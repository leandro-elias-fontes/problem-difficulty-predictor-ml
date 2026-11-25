"""Modelo semantico multilingue (PT-BR + EN) com preditor de tags, k-NN e ensemble de rating."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.dataset import clean_text
from src.features import extract_numeric_features
from src.train import RatingMLP, _train_mlp_fold

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_ARTIFACTS_DIR = Path("artifacts/semantic")

CORE_TAGS: list[str] = [
    "implementation",
    "math",
    "greedy",
    "dp",
    "data structures",
    "brute force",
    "constructive algorithms",
    "graphs",
    "sortings",
    "binary search",
    "dfs and similar",
    "trees",
    "strings",
    "number theory",
    "combinatorics",
    "two pointers",
    "bitmasks",
    "shortest paths",
]

TAG_PT_MAP: dict[str, str] = {
    "implementation": "Implementação / Lógica",
    "math": "Matemática",
    "greedy": "Algoritmo Guloso",
    "dp": "Programação Dinâmica (DP)",
    "data structures": "Estruturas de Dados",
    "brute force": "Força Bruta / Backtracking",
    "constructive algorithms": "Lógica Construtiva",
    "graphs": "Grafos",
    "sortings": "Ordenação",
    "binary search": "Busca Binária",
    "dfs and similar": "DFS / BFS",
    "trees": "Árvores",
    "strings": "Strings",
    "number theory": "Teoria dos Números",
    "combinatorics": "Combinatória",
    "two pointers": "Dois Ponteiros / Janela",
    "bitmasks": "Bitmask",
    "shortest paths": "Menor Caminho (Dijkstra/BFS)",
}


def rating_to_track(rating: float) -> dict[str, str]:
    """Mapeia o rating para faixas de treino, turmas e posicao sugerida na lista."""
    if rating < 1150:
        return {
            "level": "Iniciante",
            "track": "Turma Iniciante (Lógica & Sintaxe)",
            "slot": "Questão A–B (Aquecimento)",
            "guidance": (
                "Ideal para primeiras semanas de programação: foco em leitura de entrada, "
                "condicionais, laços e vetores simples."
            ),
        }
    if rating < 1550:
        return {
            "level": "Intermediário",
            "track": "Turma Intermediária / OBI",
            "slot": "Questão C–D (Meio de Lista)",
            "guidance": (
                "Excelente para consolidar algoritmos clássicos (ordenação, guloso, busca binária, "
                "dois ponteiros e BFS/DFS básico)."
            ),
        }
    if rating < 1950:
        return {
            "level": "Avançado",
            "track": "Turma Avançada / Maratona SBC",
            "slot": "Questão E–G (Final de Lista / Seletiva)",
            "guidance": (
                "Exige modelagem algorítmica sólida (grafos ponderados, DP clássica, árvores "
                "ou estruturas de dados O(log N))."
            ),
        }
    if rating < 2350:
        return {
            "level": "Expert (ICPC Regional)",
            "track": "Treino Regional ICPC",
            "slot": "Questão Desafio",
            "guidance": (
                "Questão de nível Regional ICPC. Recomendada como desafio extra para times "
                "que já dominam DP avançada, SegTree e árvores."
            ),
        }
    return {
        "level": "Grandmaster (ICPC Finals)",
        "track": "ICPC Finals / Desafio Extremo",
        "slot": "Questão Âncora",
        "guidance": (
            "Altíssima complexidade técnica ou matemática. Indicada apenas para treino de "
            "elite ou discussão dirigida em monitoria."
        ),
    }


@lru_cache(maxsize=1)
def get_embedder() -> object:
    """Carrega e mantem em cache o encoder multilingue PT-BR / EN."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Gera embeddings normalizados (L2=1) com o modelo multilingue."""
    model = get_embedder()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def _build_tag_matrix(tags_series: pd.Series) -> np.ndarray:
    mat = np.zeros((len(tags_series), len(CORE_TAGS)), dtype=np.int32)
    tag_to_idx = {t: i for i, t in enumerate(CORE_TAGS)}
    for row_i, raw in enumerate(tags_series.fillna("")):
        for tag in str(raw).split("|"):
            idx = tag_to_idx.get(tag.strip().lower())
            if idx is not None:
                mat[row_i, idx] = 1
    return mat


def train_semantic_pipeline(
    data_path: Path = Path("data/real_problems.csv"),
    out_dir: Path = SEMANTIC_ARTIFACTS_DIR,
    n_splits: int = 5,
    epochs: int = 30,
    seed: int = 42,
) -> dict[str, object]:
    """Treina classificador de tags, indice k-NN e ensemble (XGBoost + Ridge + PyTorch MLP)."""
    df = pd.read_csv(data_path)
    df["statement"] = df["statement"].map(clean_text)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy().reset_index(drop=True)
    df["rating"] = df["rating"].clip(800.0, 3000.0)
    if "time_limit" not in df.columns:
        df["time_limit"] = 2.0
    if "title" not in df.columns:
        df["title"] = df["statement"].str.slice(0, 65) + "..."

    print(f"Gerando embeddings multilingues para {len(df)} problemas reais...")
    emb = encode_texts(df["statement"].tolist(), batch_size=64)

    num_df = extract_numeric_features(df)
    num_raw = num_df.to_numpy(dtype=np.float32)
    y = df["rating"].to_numpy(dtype=np.float64)
    y_tags = _build_tag_matrix(df["tags"] if "tags" in df.columns else pd.Series([""] * len(df)))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_xgb = np.zeros(len(df), dtype=np.float64)
    oof_ridge = np.zeros(len(df), dtype=np.float64)
    oof_mlp = np.zeros(len(df), dtype=np.float64)
    oof_tag_probs = np.zeros((len(df), len(CORE_TAGS)), dtype=np.float32)

    print("Executando validacao cruzada 5-Fold (Stage 1: Tags -> Stage 2: Rating Ensemble)...")
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(df)):
        scaler = StandardScaler()
        num_tr = scaler.fit_transform(num_raw[tr_idx])
        num_va = scaler.transform(num_raw[va_idx])

        x_stage1_tr = np.hstack([emb[tr_idx], num_tr])
        x_stage1_va = np.hstack([emb[va_idx], num_va])

        # Stage 1: Classificador multi-label de tags reais
        tag_clf = OneVsRestClassifier(
            LogisticRegression(C=1.5, max_iter=400, solver="lbfgs"),
            n_jobs=-1,
        )
        tag_clf.fit(x_stage1_tr, y_tags[tr_idx])
        tr_tag_probs = np.asarray(tag_clf.predict_proba(x_stage1_tr), dtype=np.float32)
        va_tag_probs = np.asarray(tag_clf.predict_proba(x_stage1_va), dtype=np.float32)
        oof_tag_probs[va_idx] = va_tag_probs

        # Stage 2: Features completas = Embeddings + Probabilidades de Tags + Features Numericas
        x_tr = np.hstack([emb[tr_idx], tr_tag_probs, num_tr]).astype(np.float32)
        x_va = np.hstack([emb[va_idx], va_tag_probs, num_va]).astype(np.float32)
        y_tr, y_va = y[tr_idx], y[va_idx]

        # 2a. Ridge Regression (excelente em embeddings densos)
        ridge = Ridge(alpha=25.0)
        ridge.fit(x_tr, y_tr)
        oof_ridge[va_idx] = np.clip(ridge.predict(x_va), 800.0, 3000.0)

        # 2b. XGBoost Regressor
        xgb = XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.45,
            reg_lambda=5.0,
            random_state=seed + fold_idx,
            n_jobs=-1,
        )
        xgb.fit(x_tr, y_tr)
        oof_xgb[va_idx] = np.clip(xgb.predict(x_va), 800.0, 3000.0)

        # 2c. PyTorch MLP
        _, va_mlp, _, _, _, _ = _train_mlp_fold(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_va,
            y_val=y_va,
            epochs=epochs,
            batch_size=64,
            lr=1.5e-3,
            seed=seed + fold_idx,
        )
        oof_mlp[va_idx] = va_mlp

    oof_ens = 0.45 * oof_xgb + 0.30 * oof_ridge + 0.25 * oof_mlp
    metrics = {
        "n_samples": len(df),
        "xgboost": {
            "mae": float(mean_absolute_error(y, oof_xgb)),
            "rmse": float(root_mean_squared_error(y, oof_xgb)),
            "r2": float(r2_score(y, oof_xgb)),
        },
        "ridge": {
            "mae": float(mean_absolute_error(y, oof_ridge)),
            "rmse": float(root_mean_squared_error(y, oof_ridge)),
            "r2": float(r2_score(y, oof_ridge)),
        },
        "pytorch_mlp": {
            "mae": float(mean_absolute_error(y, oof_mlp)),
            "rmse": float(root_mean_squared_error(y, oof_mlp)),
            "r2": float(r2_score(y, oof_mlp)),
        },
        "ensemble": {
            "mae": float(mean_absolute_error(y, oof_ens)),
            "rmse": float(root_mean_squared_error(y, oof_ens)),
            "r2": float(r2_score(y, oof_ens)),
        },
    }

    # Treino final no dataset completo
    print("Treinando modelos finais no dataset completo...")
    final_scaler = StandardScaler()
    num_all = final_scaler.fit_transform(num_raw)
    x_s1_all = np.hstack([emb, num_all])

    final_tag_clf = OneVsRestClassifier(
        LogisticRegression(C=1.5, max_iter=400, solver="lbfgs"),
        n_jobs=-1,
    )
    final_tag_clf.fit(x_s1_all, y_tags)
    all_tag_probs = np.asarray(final_tag_clf.predict_proba(x_s1_all), dtype=np.float32)

    x_all = np.hstack([emb, all_tag_probs, num_all]).astype(np.float32)

    final_ridge = Ridge(alpha=25.0)
    final_ridge.fit(x_all, y)

    final_xgb = XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.45,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
    )
    final_xgb.fit(x_all, y)

    final_mlp, _, _, _, y_mean, y_std = _train_mlp_fold(
        x_train=x_all,
        y_train=y,
        x_val=x_all,
        y_val=y,
        epochs=epochs,
        batch_size=64,
        lr=1.5e-3,
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_scaler, out_dir / "num_scaler.joblib")
    joblib.dump(final_tag_clf, out_dir / "tag_clf.joblib")
    joblib.dump(final_ridge, out_dir / "ridge.joblib")
    joblib.dump(final_xgb, out_dir / "xgb.joblib")
    torch.save(
        {
            "state_dict": final_mlp.state_dict(),
            "input_dim": x_all.shape[1],
            "y_mean": y_mean,
            "y_std": y_std,
        },
        out_dir / "mlp.pt",
    )

    # Salva indice k-NN compacto (float16) apenas dos problemas reais unicos para busca de similares
    cf_mask = (
        (df["source"] == "codeforces")
        if "source" in df.columns
        else pd.Series(True, index=df.index)
    )
    knn_df = df[cf_mask].reset_index(drop=True)
    knn_emb = emb[cf_mask.to_numpy()].astype(np.float16)
    np.save(out_dir / "knn_embeddings.npy", knn_emb)
    knn_meta = [
        {
            "title": str(r["title"])[:90],
            "rating": int(round(float(r["rating"]))),
            "tags": [TAG_PT_MAP[t] for t in str(r.get("tags", "")).split("|") if t in TAG_PT_MAP][
                :3
            ],
        }
        for _, r in knn_df.iterrows()
    ]
    (out_dir / "knn_meta.json").write_text(
        json.dumps(knn_meta, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


@lru_cache(maxsize=2)
def _load_semantic_bundle(out_dir_str: str) -> dict[str, object]:
    out_dir = Path(out_dir_str)
    scaler = joblib.load(out_dir / "num_scaler.joblib")
    tag_clf = joblib.load(out_dir / "tag_clf.joblib")
    ridge = joblib.load(out_dir / "ridge.joblib")
    xgb = joblib.load(out_dir / "xgb.joblib")
    ckpt = torch.load(out_dir / "mlp.pt", map_location="cpu", weights_only=True)
    mlp = RatingMLP(input_dim=int(ckpt["input_dim"]))
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()
    knn_emb = np.load(out_dir / "knn_embeddings.npy").astype(np.float32)
    knn_meta = json.loads((out_dir / "knn_meta.json").read_text(encoding="utf-8"))
    return {
        "scaler": scaler,
        "tag_clf": tag_clf,
        "ridge": ridge,
        "xgb": xgb,
        "mlp": mlp,
        "y_mean": float(ckpt["y_mean"]),
        "y_std": float(ckpt["y_std"]),
        "knn_emb": knn_emb,
        "knn_meta": knn_meta,
    }


def semantic_artifacts_available(out_dir: Path = SEMANTIC_ARTIFACTS_DIR) -> bool:
    """Verifica se o bundle semantico multilingue ja esta treinado em disco."""
    return (out_dir / "xgb.joblib").exists() and (out_dir / "knn_embeddings.npy").exists()


def predict_semantic(
    statement: str,
    feat_df: pd.DataFrame,
    out_dir: Path = SEMANTIC_ARTIFACTS_DIR,
) -> dict[str, object]:
    """Executa inferencia semantica multilingue (Tags + Ensemble + Top-3 Similares)."""
    bundle = _load_semantic_bundle(str(out_dir.resolve()))
    emb = encode_texts([statement], batch_size=1)
    num_raw = feat_df.drop(columns=["statement"], errors="ignore").to_numpy(dtype=np.float32)
    num_scaled = bundle["scaler"].transform(num_raw)

    x_s1 = np.hstack([emb, num_scaled])
    tag_probs = np.asarray(bundle["tag_clf"].predict_proba(x_s1), dtype=np.float32)[0]

    # Top tags previstas pelo classificador neural/linear
    ranked_tags = sorted(zip(CORE_TAGS, tag_probs, strict=True), key=lambda x: x[1], reverse=True)
    predicted_topics = [TAG_PT_MAP[t] for t, p in ranked_tags if p >= 0.28 and t in TAG_PT_MAP][:4]
    if not predicted_topics:
        predicted_topics = [TAG_PT_MAP[ranked_tags[0][0]]]

    x_s2 = np.hstack([emb, tag_probs.reshape(1, -1), num_scaled]).astype(np.float32)
    xgb_pred = float(np.clip(bundle["xgb"].predict(x_s2)[0], 800.0, 3000.0))
    ridge_pred = float(np.clip(bundle["ridge"].predict(x_s2)[0], 800.0, 3000.0))
    with torch.no_grad():
        raw_mlp = float(bundle["mlp"](torch.tensor(x_s2, dtype=torch.float32)).item())
    mlp_pred = float(np.clip(raw_mlp * bundle["y_std"] + bundle["y_mean"], 800.0, 3000.0))

    # Busca k-NN de problemas reais similares (cosseno via produto interno)
    sims = bundle["knn_emb"] @ emb[0]
    top_idx = np.argsort(sims)[-3:][::-1]
    similar_problems: list[dict[str, object]] = []
    sim_ratings: list[float] = []
    sim_weights: list[float] = []
    for idx in top_idx:
        sim_score = float(sims[idx])
        meta = bundle["knn_meta"][int(idx)]
        similar_problems.append(
            {
                "title": meta["title"],
                "rating": meta["rating"],
                "tags": meta["tags"],
                "similarity": int(round(max(sim_score, 0.0) * 100)),
            }
        )
        if sim_score >= 0.55:
            sim_ratings.append(float(meta["rating"]))
            sim_weights.append(sim_score**2)

    ens_pred = 0.45 * xgb_pred + 0.30 * ridge_pred + 0.25 * mlp_pred
    # Se encontrou problemas muito parecidos na base real, aplica ancora suave
    if sim_weights and max(sim_weights) >= 0.45:
        knn_anchor = float(np.average(sim_ratings, weights=sim_weights))
        ens_pred = 0.80 * ens_pred + 0.20 * knn_anchor

    rounded = int(round(ens_pred / 50.0) * 50)
    spread = float(np.std([xgb_pred, ridge_pred, mlp_pred]))
    confidence = "Alta" if spread <= 110 else ("Média" if spread <= 220 else "Baixa")
    track_info = rating_to_track(rounded)

    return {
        "rating": rounded,
        "level": track_info["level"],
        "track": track_info["track"],
        "slot": track_info["slot"],
        "guidance": track_info["guidance"],
        "topics": predicted_topics,
        "confidence": confidence,
        "xgb_rating": round(xgb_pred, 1),
        "ridge_rating": round(ridge_pred, 1),
        "mlp_rating": round(mlp_pred, 1),
        "similar_problems": similar_problems,
    }


if __name__ == "__main__":
    train_semantic_pipeline()
