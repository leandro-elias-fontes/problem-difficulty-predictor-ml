"""CLI simples para estimar o Rating de um problema (texto ou imagem PNG/JPG/WEBP)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.dataset import clean_text, rating_to_level
from src.features import build_feature_frame
from src.train import RatingMLP, train_and_evaluate
from src.vision_extract import extract_problems_from_image


def predict_rating(
    statement: str,
    n_limit: str | None = None,
    time_limit: float = 2.0,
    artifacts_dir: Path = Path("artifacts"),
) -> dict[str, object]:
    """Estima o Rating usando XGBoost e PyTorch MLP treinados."""
    xgb_path = artifacts_dir / "xgb_pipeline.joblib"
    if not xgb_path.exists():
        train_and_evaluate(artifacts_dir=artifacts_dir)

    full_statement = clean_text(statement)
    if n_limit:
        full_statement = f"{full_statement} Limite: n <= {n_limit}."

    row_df = pd.DataFrame([{"statement": full_statement, "time_limit": float(time_limit)}])
    feat_df = build_feature_frame(row_df)

    xgb_pipe = joblib.load(xgb_path)
    xgb_pred = float(np.clip(xgb_pipe.predict(feat_df)[0], 800.0, 3000.0))

    mlp_prep = joblib.load(artifacts_dir / "mlp_preprocessor.joblib")
    ckpt = torch.load(artifacts_dir / "mlp_model.pt", map_location="cpu", weights_only=True)
    model = RatingMLP(input_dim=int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_mat = mlp_prep.transform(feat_df)
    if hasattr(x_mat, "toarray"):
        x_mat = x_mat.toarray()
    with torch.no_grad():
        raw_mlp = float(model(torch.tensor(x_mat, dtype=torch.float32)).item())
    mlp_pred = float(np.clip(raw_mlp * float(ckpt["y_std"]) + float(ckpt["y_mean"]), 800.0, 3000.0))

    avg_pred = 0.5 * (xgb_pred + mlp_pred)
    rounded = int(round(avg_pred / 50.0) * 50)

    return {
        "rating": rounded,
        "level": rating_to_level(rounded),
        "xgb_rating": round(xgb_pred, 1),
        "mlp_rating": round(mlp_pred, 1),
        "max_n_log10": round(float(feat_df["max_n_log10"].iloc[0]), 2),
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
