"""CLI simples para estimar o Rating de um problema (texto ou imagem PNG/JPG/WEBP)."""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.dataset import clean_text, rating_to_level
from src.features import (
    build_feature_frame,
    extract_detected_topics,
    extract_max_n_log10,
)
from src.llm_coach import analyze_with_local_llm
from src.semantic_model import (
    SEMANTIC_ARTIFACTS_DIR,
    predict_semantic,
    rating_to_track,
    semantic_artifacts_available,
)
from src.train import RatingMLP, train_and_evaluate
from src.vision_extract import extract_problems_from_image


@lru_cache(maxsize=4)
def _load_artifacts(artifacts_dir_str: str) -> tuple[object, object, RatingMLP, float, float]:
    """Carrega e mantem em cache os modelos treinados para inferencia rapida."""
    artifacts_dir = Path(artifacts_dir_str)
    xgb_path = artifacts_dir / "xgb_pipeline.joblib"
    if not xgb_path.exists():
        train_and_evaluate(artifacts_dir=artifacts_dir)

    xgb_pipe = joblib.load(xgb_path)
    mlp_prep = joblib.load(artifacts_dir / "mlp_preprocessor.joblib")
    ckpt = torch.load(artifacts_dir / "mlp_model.pt", map_location="cpu", weights_only=True)
    model = RatingMLP(input_dim=int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return xgb_pipe, mlp_prep, model, float(ckpt["y_mean"]), float(ckpt["y_std"])


def _format_n_display(log10_val: float) -> str:
    """Formata log10(N) em notacao legivel para instrutores."""
    if abs(log10_val - 2.5) < 1e-6:
        return "Não informado"
    rounded_exp = round(log10_val)
    if abs(log10_val - rounded_exp) < 0.08 and rounded_exp >= 3:
        return f"N ≤ 10^{rounded_exp}"
    approx = 10**log10_val
    if approx < 10_000:
        return f"N ≤ {int(round(approx))}"
    return f"N ≈ 10^{log10_val:.1f}"


def _instructor_guidance(rating: int) -> str:
    """Retorna orientacao pratica para montagem de listas e provas."""
    if rating < 1000:
        return "Questão de aquecimento (A/B). Ideal para primeiras semanas ou início de lista."
    if rating < 1250:
        return "Nível Fácil. Boa para fixar lógica básica, vetores e ordenação."
    if rating < 1600:
        return "Nível Intermediário. Ótima para meio de lista (C/D) com algoritmos clássicos."
    if rating < 2000:
        return "Nível Avançado. Recomendada para final de lista ou seletiva de maratona."
    if rating < 2400:
        return "Nível Difícil (Regional ICPC). Ideal como questão desafio da prova."
    return "Nível Muito Difícil (ICPC Finals). Questão âncora para times experientes."


def predict_rating(
    statement: str,
    n_limit: str | None = None,
    time_limit: float = 2.0,
    artifacts_dir: Path = Path("artifacts"),
) -> dict[str, object]:
    """Estima o Rating usando o modelo semantico multilingue (ou fallback TF-IDF)."""
    full_statement = clean_text(statement)
    if n_limit:
        full_statement = f"{full_statement} Limite: n <= {n_limit}."

    row_df = pd.DataFrame([{"statement": full_statement, "time_limit": float(time_limit)}])
    feat_df = build_feature_frame(row_df)

    # Se o usuario informou um limite explicito de N, prioriza esse valor sobre numeros do texto
    if n_limit:
        explicit_log10 = extract_max_n_log10(f"n <= {n_limit}", default_log10=-1.0)
        if explicit_log10 >= 0.0:
            idx0 = feat_df.index[0]
            feat_df.loc[idx0, "max_n_log10"] = explicit_log10
            has_signal = (
                feat_df.loc[idx0, "has_counting"] > 0
                or feat_df.loc[idx0, "has_dp"] > 0
                or feat_df.loc[idx0, "has_graph"] > 0
            )
            feat_df.loc[idx0, "is_small_n_exp"] = float(explicit_log10 <= 1.45 and has_signal)
            feat_df.loc[idx0, "is_large_n_nlogn"] = float(4.5 <= explicit_log10 <= 6.5)
            feat_df.loc[idx0, "is_huge_n_log"] = float(explicit_log10 >= 8.5)

    max_n_log10 = round(float(feat_df["max_n_log10"].iloc[0]), 2)
    n_display = _format_n_display(max_n_log10)

    low_mem = os.getenv("LOW_MEMORY_MODE", "0").strip().lower() in {"1", "true", "yes"}
    sem_dir = artifacts_dir / "semantic"
    active_sem_dir = None
    if not low_mem:
        active_sem_dir = (
            sem_dir
            if semantic_artifacts_available(sem_dir)
            else (
                SEMANTIC_ARTIFACTS_DIR
                if artifacts_dir == Path("artifacts")
                and semantic_artifacts_available(SEMANTIC_ARTIFACTS_DIR)
                else None
            )
        )
    if active_sem_dir is not None:
        try:
            sem_res = predict_semantic(full_statement, feat_df, out_dir=active_sem_dir)
        except ImportError:
            sem_res = None
        if sem_res is not None:
            sem_rating = int(sem_res["rating"])
            sem_topics = list(sem_res["topics"])

            llm_res = analyze_with_local_llm(
                statement=full_statement,
                n_display=n_display,
                time_limit=float(time_limit),
                semantic_rating=sem_rating,
                semantic_topics=sem_topics,
            )
            if llm_res is not None:
                blended = 0.65 * float(llm_res["llm_rating"]) + 0.35 * float(sem_rating)
                final_rating = int(round(blended / 50.0) * 50)
                track_info = rating_to_track(final_rating)
                merged_topics: list[str] = []
                for t in [*list(llm_res["topics"]), *sem_topics]:
                    if t not in merged_topics:
                        merged_topics.append(t)
                provider = str(llm_res.get("provider") or "Coach IA")
                return {
                    **sem_res,
                    "rating": final_rating,
                    "level": track_info["level"],
                    "track": track_info["track"],
                    "slot": track_info["slot"],
                    "guidance": track_info["guidance"],
                    "topics": merged_topics[:4],
                    "llm_rating": llm_res["llm_rating"],
                    "algorithm": llm_res["algorithm"],
                    "complexity": llm_res["complexity"],
                    "mentor_tip": llm_res["mentor_tip"],
                    "ai_engine": f"{provider} + Ensemble",
                    "max_n_log10": max_n_log10,
                    "n_display": n_display,
                }

            return {
                **sem_res,
                "ai_engine": "MiniLM + Ensemble",
                "max_n_log10": max_n_log10,
                "n_display": n_display,
            }

    xgb_pipe, mlp_prep, model, y_mean, y_std = _load_artifacts(str(artifacts_dir.resolve()))
    xgb_pred = float(np.clip(xgb_pipe.predict(feat_df)[0], 800.0, 3000.0))

    x_mat = mlp_prep.transform(feat_df)
    if hasattr(x_mat, "toarray"):
        x_mat = x_mat.toarray()
    with torch.no_grad():
        raw_mlp = float(model(torch.tensor(x_mat, dtype=torch.float32)).item())
    mlp_pred = float(np.clip(raw_mlp * y_std + y_mean, 800.0, 3000.0))

    avg_pred = 0.5 * (xgb_pred + mlp_pred)
    rounded = int(round(avg_pred / 50.0) * 50)
    spread = abs(xgb_pred - mlp_pred)
    confidence = "Alta" if spread <= 120 else ("Média" if spread <= 250 else "Baixa")
    base_topics = extract_detected_topics(feat_df.iloc[0])

    llm_res = analyze_with_local_llm(
        statement=full_statement,
        n_display=n_display,
        time_limit=float(time_limit),
        semantic_rating=rounded,
        semantic_topics=base_topics,
    )
    if llm_res is not None:
        blended = 0.65 * float(llm_res["llm_rating"]) + 0.35 * float(rounded)
        rounded = int(round(blended / 50.0) * 50)
        track_info = rating_to_track(rounded)
        merged_topics: list[str] = []
        for t in [*list(llm_res["topics"]), *base_topics]:
            if t not in merged_topics:
                merged_topics.append(t)
        provider = str(llm_res.get("provider") or "Coach IA")
        return {
            "rating": rounded,
            "level": rating_to_level(rounded),
            "track": track_info["track"],
            "slot": track_info["slot"],
            "xgb_rating": round(xgb_pred, 1),
            "mlp_rating": round(mlp_pred, 1),
            "llm_rating": llm_res["llm_rating"],
            "algorithm": llm_res["algorithm"],
            "complexity": llm_res["complexity"],
            "mentor_tip": llm_res["mentor_tip"],
            "ai_engine": f"{provider} + ML Local",
            "max_n_log10": max_n_log10,
            "n_display": n_display,
            "topics": merged_topics[:4],
            "confidence": confidence,
            "guidance": track_info["guidance"],
            "similar_problems": [],
        }

    track_info = rating_to_track(rounded)
    return {
        "rating": rounded,
        "level": rating_to_level(rounded),
        "track": track_info["track"],
        "slot": track_info["slot"],
        "xgb_rating": round(xgb_pred, 1),
        "mlp_rating": round(mlp_pred, 1),
        "ai_engine": "XGBoost + PyTorch MLP (512MB Slim)",
        "max_n_log10": max_n_log10,
        "n_display": n_display,
        "topics": base_topics,
        "confidence": confidence,
        "guidance": track_info["guidance"],
        "similar_problems": [],
    }


def main() -> None:
    """Entrypoint CLI."""
    parser = argparse.ArgumentParser(
        description="Preve o Rating de um problema (via --text ou --image PNG/JPG/WEBP)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Resumo ou enunciado do problema.")
    group.add_argument(
        "--image",
        "--file",
        dest="image",
        type=Path,
        help="Caminho de imagem (PNG/JPG/WEBP) da questao ou prova.",
    )
    parser.add_argument("--n-limit", default=None, help="Limite de N (ex: 10^5, 20, 10^{18}).")
    parser.add_argument("--time-limit", type=float, default=2.0, help="Tempo limite em segundos.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    if args.image is not None:
        problems = extract_problems_from_image(args.image)
        if not problems:
            print("Nenhum texto/questao identificada na imagem.")
            return
        print(f"Questoes identificadas via OCR offline em {args.image.name}: {len(problems)}\n")
        for prob in problems:
            res = predict_rating(
                statement=prob.statement_text,
                n_limit=args.n_limit,
                time_limit=prob.time_limit,
                artifacts_dir=args.artifacts_dir,
            )
            print(f"[{prob.label}]")
            print(f"  Rating estimado: {res['rating']} - {res['level']}")
            print(
                f"  Detalhes: XGBoost={res['xgb_rating']} | PyTorch MLP={res['mlp_rating']} "
                f"| log10(N)={res['max_n_log10']} | time_limit={prob.time_limit}s\n"
            )
        return

    res = predict_rating(
        statement=args.text,
        n_limit=args.n_limit,
        time_limit=args.time_limit,
        artifacts_dir=args.artifacts_dir,
    )
    print(f"Rating estimado: {res['rating']} - {res['level']}")
    print(
        f"Detalhes: XGBoost={res['xgb_rating']} | PyTorch MLP={res['mlp_rating']} "
        f"| log10(N)={res['max_n_log10']}"
    )


if __name__ == "__main__":
    main()
