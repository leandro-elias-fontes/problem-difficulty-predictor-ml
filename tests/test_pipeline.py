"""Testes unitarios e de integracao para o pipeline de previsao de Rating."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.dataset import generate_problems_sample, load_dataset
from src.features import (
    NUMERIC_FEATURE_COLS,
    build_feature_frame,
    build_preprocessor,
    extract_max_n_log10,
    extract_numeric_features,
)
from src.train import RatingMLP, train_and_evaluate
from src.vision_extract import extract_problems_from_image, split_problems_from_ocr


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Limite: 1 <= n <= 2 * 10^5", math.log10(200_000)),
        ("Restricao: 1 <= n <= 10^{18}", 18.0),
        ("Para n <= 20 vertices", math.log10(20)),
        ("1 <= n <= 10^5 modulo 10^9 + 7", 5.0),
        ("Sem limites numericos", 2.5),
    ],
)
def test_extract_max_n_log10(text: str, expected: float) -> None:
    """Valida extracao de log10(N) e protecao contra constante 10^9 + 7."""
    assert extract_max_n_log10(text) == pytest.approx(expected, rel=1e-3)


def test_keyword_flags() -> None:
    """Valida extracao de flags de grafo, arvore, matriz e modulo."""
    df = pd.DataFrame(
        [
            {
                "statement": "Dada uma arvore com n vertices, responda modulo 10^9 + 7.",
                "time_limit": 2.5,
            },
            {
                "statement": "Encontre caminho minimo no grafo em uma matriz grid.",
                "time_limit": 1.0,
            },
        ]
    )
    feats = extract_numeric_features(df)
    assert list(feats.columns) == NUMERIC_FEATURE_COLS
    assert feats.loc[0, "has_tree"] == 1.0
    assert feats.loc[0, "has_modulo"] == 1.0
    assert feats.loc[1, "has_graph"] == 1.0
    assert feats.loc[1, "has_matrix"] == 1.0


def test_rating_mlp_shape_and_grad() -> None:
    """Verifica shape de saida e fluxo de gradiente da RatingMLP."""
    model = RatingMLP(input_dim=32)
    x = torch.randn(6, 32)
    out = model(x)
    assert out.shape == (6,)
    out.pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_no_data_leakage_in_preprocessor() -> None:
    """Garante que TF-IDF e StandardScaler sao ajustados apenas no treino."""
    train_df = pd.DataFrame(
        [
            {"statement": "grafo caminho 1 <= n <= 10", "time_limit": 1.0},
            {"statement": "arvore dp 1 <= n <= 100", "time_limit": 2.0},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"statement": "tokenexclusivoteste 1 <= n <= 10^18", "time_limit": 50.0},
        ]
    )

    x_tr = build_feature_frame(train_df)
    x_te = build_feature_frame(test_df)
    prep = build_preprocessor(max_tfidf_features=50)
    prep.fit(x_tr)
    transformed = prep.transform(x_te)
    assert transformed.shape[0] == 1

    vocab = prep.named_transformers_["text"].vocabulary_
    assert "tokenexclusivoteste" not in vocab

    scaler = prep.named_transformers_["num"].named_steps["scaler"]
    time_idx = NUMERIC_FEATURE_COLS.index("time_limit_sec")
    assert scaler.mean_[time_idx] == pytest.approx(1.5)


def test_dataset_and_train_smoke(tmp_path: Path) -> None:
    """Valida gerador de 300 problemas e smoke test do KFold (XGBoost + PyTorch)."""
    csv_path = tmp_path / "problems_sample.csv"
    generate_problems_sample(output_path=csv_path, n_samples=300, seed=42)
    df = load_dataset(csv_path=csv_path, generate_if_missing=False)
    assert len(df) == 300
    assert df["rating"].between(800, 2800).all()

    summary = train_and_evaluate(
        data_path=csv_path,
        assets_dir=tmp_path / "assets",
        artifacts_dir=tmp_path / "artifacts",
        n_splits=2,
        epochs=3,
        max_tfidf_features=60,
        seed=42,
    )
    assert summary.xgb_metrics["r2"] > 0.3
    assert (tmp_path / "assets" / "loss_curve.png").exists()
    assert (tmp_path / "assets" / "pred_vs_actual.png").exists()


def test_vision_extract_ocr_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Valida separacao de questoes por OCR offline e rejeicao de formato invalido."""
    ocr_sample = (
        "Problem A: Soma Simples\n"
        "Time limit: 1.0 seconds\n"
        "Dado um array, 1 <= n <= 100.\n"
        "Problem B: Caminhos em Arvore\n"
        "Time limit: 2.5 seconds\n"
        "Dada uma arvore, 1 <= n <= 10^5."
    )
    problems = split_problems_from_ocr(ocr_sample)
    assert len(problems) == 2
    assert problems[0].time_limit == pytest.approx(1.0)
    assert problems[1].time_limit == pytest.approx(2.5)

    img_path = tmp_path / "questao.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    monkeypatch.setattr("src.vision_extract.extract_text_from_image", lambda _: ocr_sample)
    extracted = extract_problems_from_image(img_path)
    assert len(extracted) == 2

    bad_path = tmp_path / "prova.pdf"
    bad_path.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="Formato nao suportado"):
        extract_problems_from_image(bad_path)


def test_fastapi_web_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valida interface web e endpoints /health, /api/predict-text e /api/predict-image."""
    from fastapi.testclient import TestClient

    from src.app import app
    from src.vision_extract import ExtractedProblem

    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}

    resp = client.post(
        "/api/predict-text",
        json={
            "statement": "Dada uma arvore com n vertices, calcule subarvores.",
            "n_limit": "10^5",
            "time_limit": 2.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 800 <= data["rating"] <= 3000

    monkeypatch.setattr(
        "src.app.extract_problems_from_image",
        lambda _: [
            ExtractedProblem(
                label="Problem A",
                statement_text="Dado um grafo conexo, 1 <= n <= 10^5.",
                time_limit=2.0,
            )
        ],
    )
    img_resp = client.post(
        "/api/predict-image",
        files={"file": ("questao.png", b"\x89PNG\r\n\x1a\nfake", "image/png")},
    )
    assert img_resp.status_code == 200
    img_data = img_resp.json()
    assert img_data["count"] == 1
    assert 800 <= img_data["results"][0]["rating"] <= 3000
