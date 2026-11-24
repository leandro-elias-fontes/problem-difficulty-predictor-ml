"""Treinamento e comparacao via KFold: XGBoost Regressor vs PyTorch MLP."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.dataset import load_dataset
from src.features import build_feature_frame, build_preprocessor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class RatingMLP(nn.Module):
    """MLP simples em PyTorch sobre features TF-IDF + restricoes matematicas."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna predicao normalizada de shape (batch_size,)."""
        return self.net(x).squeeze(-1)


@dataclass
class TrainSummary:
    """Resultados da validacao cruzada KFold para XGBoost e PyTorch MLP."""

    xgb_metrics: dict[str, float]
    mlp_metrics: dict[str, float]
    train_losses: list[float]
    val_losses: list[float]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE, RMSE e R2."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _train_mlp_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[RatingMLP, np.ndarray, list[float], list[float], float, float]:
    """Treina a RatingMLP em um fold com normalizacao do alvo apenas no treino."""
    torch.manual_seed(seed)
    y_mean = float(np.mean(y_train))
    y_std = float(max(np.std(y_train), 1e-6))

    y_tr_norm = (y_train - y_mean) / y_std
    y_va_norm = (y_val - y_mean) / y_std

    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_tr_norm, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = RatingMLP(input_dim=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.MSELoss()

    x_va_t = torch.tensor(x_val, dtype=torch.float32)
    y_va_t = torch.tensor(y_va_norm, dtype=torch.float32)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        total = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        train_losses.append(total / max(len(loader), 1))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_va_t), y_va_t).item())
        val_losses.append(val_loss)

    model.eval()
    with torch.no_grad():
        raw_preds = model(x_va_t).numpy()
    val_preds = np.clip(raw_preds * y_std + y_mean, 800.0, 3000.0)
    return model, val_preds, train_losses, val_losses, y_mean, y_std


def plot_loss_curve(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    """Salva grafico de Train Loss vs Val Loss da MLP em PyTorch."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs, train_losses, label="Train Loss (MSE)", linewidth=2.0, color="#1f77b4")
    ax.plot(epochs, val_losses, label="Val Loss (MSE)", linewidth=2.0, color="#d62728")
    ax.set_title("PyTorch MLP: Train vs Validation Loss (Media 5-Fold)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (Normalizada)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pred_vs_actual(
    y_true: np.ndarray,
    xgb_preds: np.ndarray,
    mlp_preds: np.ndarray,
    xgb_metrics: dict[str, float],
    mlp_metrics: dict[str, float],
    out_path: Path,
) -> None:
    """Salva grafico comparando Rating Real vs Predito (XGBoost e PyTorch MLP)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(
        y_true,
        xgb_preds,
        alpha=0.5,
        s=26,
        color="#ff7f0e",
        label=f"XGBoost (MAE={xgb_metrics['mae']:.1f}, R2={xgb_metrics['r2']:.2f})",
    )
    ax.scatter(
        y_true,
        mlp_preds,
        alpha=0.5,
        s=26,
        color="#2ca02c",
        label=f"PyTorch MLP (MAE={mlp_metrics['mae']:.1f}, R2={mlp_metrics['r2']:.2f})",
    )
    lims = [750, 2900]
    ax.plot(lims, lims, "--", color="#333333", linewidth=1.5, label="Ideal (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Rating Real vs Predito (KFold Out-of-Fold)")
    ax.set_xlabel("Rating Real")
    ax.set_ylabel("Rating Predito")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def train_and_evaluate(
    data_path: Path | None = None,
    assets_dir: Path = Path("assets"),
    artifacts_dir: Path = Path("artifacts"),
    n_splits: int = 5,
    epochs: int = 40,
    max_tfidf_features: int = 400,
    seed: int = 42,
) -> TrainSummary:
    """Executa KFold CV comparando XGBoost e PyTorch MLP, gera graficos e salva artefatos."""
    df = load_dataset(csv_path=data_path, generate_if_missing=True, seed=seed)
    x_df = build_feature_frame(df)
    y = df["rating"].to_numpy(dtype=np.float64)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_xgb = np.zeros(len(df), dtype=np.float64)
    oof_mlp = np.zeros(len(df), dtype=np.float64)
    fold_tr_losses: list[list[float]] = []
    fold_va_losses: list[list[float]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(x_df)):
        x_tr, x_va = x_df.iloc[tr_idx], x_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # 1. XGBoost Regressor
        xgb_pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor(max_tfidf_features=max_tfidf_features)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.06,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        xgb_pipe.fit(x_tr, y_tr)
        oof_xgb[va_idx] = np.clip(xgb_pipe.predict(x_va), 800.0, 3000.0)

        # 2. PyTorch MLP (pre-processador ajustado estritamente no fold de treino)
        mlp_prep = build_preprocessor(max_tfidf_features=max_tfidf_features)
        x_tr_mat = mlp_prep.fit_transform(x_tr)
        x_va_mat = mlp_prep.transform(x_va)
        if hasattr(x_tr_mat, "toarray"):
            x_tr_mat = x_tr_mat.toarray()
            x_va_mat = x_va_mat.toarray()

        _, val_preds, tr_loss, va_loss, _, _ = _train_mlp_fold(
            x_train=np.asarray(x_tr_mat, dtype=np.float32),
            y_train=y_tr,
            x_val=np.asarray(x_va_mat, dtype=np.float32),
            y_val=y_va,
            epochs=epochs,
            batch_size=32,
            lr=3e-3,
            seed=seed + fold_idx,
        )
        oof_mlp[va_idx] = val_preds
        fold_tr_losses.append(tr_loss)
        fold_va_losses.append(va_loss)

    xgb_metrics = compute_metrics(y, oof_xgb)
    mlp_metrics = compute_metrics(y, oof_mlp)
    mean_tr_loss = np.mean(np.array(fold_tr_losses), axis=0).tolist()
    mean_va_loss = np.mean(np.array(fold_va_losses), axis=0).tolist()

    plot_loss_curve(mean_tr_loss, mean_va_loss, assets_dir / "loss_curve.png")
    plot_pred_vs_actual(
        y,
        oof_xgb,
        oof_mlp,
        xgb_metrics,
        mlp_metrics,
        assets_dir / "pred_vs_actual.png",
    )

    # Treina modelos finais no dataset completo para inferencia via CLI
    final_xgb = Pipeline(
        steps=[
            ("prep", build_preprocessor(max_tfidf_features=max_tfidf_features)),
            (
                "model",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.06,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    final_xgb.fit(x_df, y)

    final_mlp_prep = build_preprocessor(max_tfidf_features=max_tfidf_features)
    x_all = final_mlp_prep.fit_transform(x_df)
    if hasattr(x_all, "toarray"):
        x_all = x_all.toarray()
    x_all_arr = np.asarray(x_all, dtype=np.float32)

    final_mlp, _, _, _, y_mean, y_std = _train_mlp_fold(
        x_train=x_all_arr,
        y_train=y,
        x_val=x_all_arr,
        y_val=y,
        epochs=epochs,
        batch_size=32,
        lr=3e-3,
        seed=seed,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_xgb, artifacts_dir / "xgb_pipeline.joblib")
    joblib.dump(final_mlp_prep, artifacts_dir / "mlp_preprocessor.joblib")
    torch.save(
        {
            "state_dict": final_mlp.state_dict(),
            "input_dim": x_all_arr.shape[1],
            "y_mean": y_mean,
            "y_std": y_std,
        },
        artifacts_dir / "mlp_model.pt",
    )
    (artifacts_dir / "metrics.json").write_text(
        json.dumps({"xgboost": xgb_metrics, "pytorch_mlp": mlp_metrics}, indent=2),
        encoding="utf-8",
    )

    return TrainSummary(
        xgb_metrics=xgb_metrics,
        mlp_metrics=mlp_metrics,
        train_losses=mean_tr_loss,
        val_losses=mean_va_loss,
    )


def main() -> None:
    """Entrypoint CLI para treino e comparacao dos modelos."""
    parser = argparse.ArgumentParser(description="Treina XGBoost e PyTorch MLP via KFold.")
    parser.add_argument("--data", type=Path, default=None, help="Caminho para CSV de problemas.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = train_and_evaluate(
        data_path=args.data,
        n_splits=args.folds,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(
        json.dumps(
            {"xgboost": summary.xgb_metrics, "pytorch_mlp": summary.mlp_metrics},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
