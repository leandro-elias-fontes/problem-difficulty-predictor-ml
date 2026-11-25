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
        ("Limite: 1 <= n <= 10⁵ e até 100.000 consultas", 5.0),
        ("Sem limites numericos", 2.5),
    ],
)
def test_extract_max_n_log10(text: str, expected: float) -> None:
    """Valida extracao de log10(N) e protecao contra constante 10^9 + 7."""
    assert extract_max_n_log10(text) == pytest.approx(expected, rel=1e-3)


def test_keyword_flags() -> None:
    """Valida extracao de flags com e sem acentos (grafo, arvore, busca binaria, DP)."""
    df = pd.DataFrame(
        [
            {
                "statement": "Dada uma árvore com n vértices, responda módulo 10^9 + 7.",
                "time_limit": 2.5,
            },
            {
                "statement": "Encontre caminho mínimo no grafo em uma matriz grid.",
                "time_limit": 1.0,
            },
            {
                "statement": "Use busca binária e programação dinâmica na subsequência.",
                "time_limit": 2.0,
            },
        ]
    )
    feats = extract_numeric_features(df)
    assert list(feats.columns) == NUMERIC_FEATURE_COLS
    assert feats.loc[0, "has_tree"] == 1.0
    assert feats.loc[0, "has_modulo"] == 1.0
    assert feats.loc[1, "has_graph"] == 1.0
    assert feats.loc[1, "has_matrix"] == 1.0
    assert feats.loc[2, "has_greedy_bs"] == 1.0
    assert feats.loc[2, "has_dp"] == 1.0


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
    monkeypatch.setenv("ENABLE_LOCAL_LLM", "0")
    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}

    resp = client.post(
        "/api/predict-text",
        json={
            "statement": "Dada uma árvore com n vértices, calcule subárvores.",
            "n_limit": "10^5",
            "time_limit": 2.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 800 <= data["rating"] <= 3000
    assert "topics" in data and len(data["topics"]) >= 1
    assert "guidance" in data
    assert data["confidence"] in {"Alta", "Média", "Baixa"}

    empty_resp = client.post(
        "/api/predict-image",
        files={"file": ("vazio.png", b"", "image/png")},
    )
    assert empty_resp.status_code == 400

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

    monkeypatch.setattr(
        "src.app.lookup_problem_by_url",
        lambda _: {
            "mode": "official",
            "source": "Codeforces",
            "title": "CF 1850E - Cardboard for Pictures",
            "rating": 1100,
            "level": "Iniciante",
            "topics": ["Busca Binária"],
            "confidence": "Oficial (Codeforces)",
            "n_display": "Rating oficial",
            "xgb_rating": 1100.0,
            "mlp_rating": 1100.0,
            "max_n_log10": 0.0,
        },
    )
    url_resp = client.post(
        "/api/predict-url",
        json={"url": "https://codeforces.com/problemset/problem/1850/E"},
    )
    assert url_resp.status_code == 200
    assert url_resp.json()["rating"] == 1100
    assert "guidance" in url_resp.json()

    fb_resp = client.post(
        "/api/feedback",
        json={"rating": 1500, "verdict": "ok", "title": "Teste", "statement_preview": "Enunciado"},
    )
    assert fb_resp.status_code == 200
    assert fb_resp.json() == {"status": "recorded"}

    batch_resp = client.post(
        "/api/analyze-batch",
        json={
            "items": [
                "https://codeforces.com/problemset/problem/1850/E",
                "Dado um grafo conexo com n vertices, calcule o menor caminho. 1 <= n <= 10^5.",
            ]
        },
    )
    assert batch_resp.status_code == 200
    batch_data = batch_resp.json()
    assert batch_data["count"] == 2
    assert batch_data["items"][0]["rating"] <= batch_data["items"][1]["rating"]


def test_match_known_problem_by_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valida identificacao automatica de questao oficial por titulo ou ID no texto/OCR."""
    from src.url_fetcher import match_known_problem_by_text

    monkeypatch.setattr(
        "src.url_fetcher._load_cf_catalog",
        lambda: {
            (4, "A"): {"name": "Watermelon", "rating": 800, "tags": ["brute force", "math"]},
            (1850, "E"): {
                "name": "Cardboard for Pictures",
                "rating": 1100,
                "tags": ["binary search"],
            },
        },
    )
    by_id = match_known_problem_by_text("1850E\nDetermine the value of w.")
    assert by_id is not None
    assert by_id["rating"] == 1100

    by_title = match_known_problem_by_text("A. Watermelon\nOne hot summer day...")
    assert by_title is not None
    assert by_title["rating"] == 800


def test_llm_coach_json_parsing_and_blend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valida parser JSON do Coach IA local e integracao com predict_rating."""
    from src.llm_coach import _parse_llm_json
    from src.predict import predict_rating

    parsed = _parse_llm_json(
        '```json\n{"rating": 1620, "algorithm": "Dijkstra", '
        '"complexity": "O(M log N)", "topics": ["Grafos"], '
        '"mentor_tip": "Modele as cidades como grafo."}\n```'
    )
    assert parsed is not None
    assert parsed["llm_rating"] == 1600
    assert parsed["algorithm"] == "Dijkstra"

    monkeypatch.setattr(
        "src.predict.analyze_with_local_llm",
        lambda **_: {**parsed, "provider": "Qwen2.5-Coder-3B (Local)"},
    )
    res = predict_rating("Um carteiro precisa ir da cidade 1 até N.", n_limit="10^5")
    assert "Qwen2.5-Coder-3B (Local)" in str(res["ai_engine"])
    assert res["algorithm"] == "Dijkstra"
    assert res["complexity"] == "O(M log N)"
    assert 800 <= int(res["rating"]) <= 3000


def test_security_headers_ssrf_and_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valida headers de seguranca, bloqueio anti-SSRF, payload e rate limit."""
    from fastapi.testclient import TestClient

    from src.app import _IP_HITS, app
    from src.url_fetcher import _validate_public_url

    _IP_HITS.clear()
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert home.headers["x-content-type-options"] == "nosniff"
    assert home.headers["x-frame-options"] == "DENY"
    assert home.headers["referrer-policy"] == "strict-origin-when-cross-origin"

    for bad_url in (
        "http://127.0.0.1:8000/health",
        "http://localhost/admin",
        "http://169.254.169.254/latest/meta-data/",
        "ftp://example.com/file",
    ):
        with pytest.raises(ValueError):
            _validate_public_url(bad_url)

    ssrf_resp = client.post("/api/predict-url", json={"url": "http://127.0.0.1:8000/health"})
    assert ssrf_resp.status_code == 400

    blank_resp = client.post("/api/predict-text", json={"statement": "   "})
    assert blank_resp.status_code == 400

    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    _IP_HITS.clear()
    assert client.post("/api/feedback", json={"rating": 1200, "verdict": "ok"}).status_code == 200
    assert client.post("/api/feedback", json={"rating": 1200, "verdict": "ok"}).status_code == 200
    assert client.post("/api/feedback", json={"rating": 1200, "verdict": "ok"}).status_code == 429
    _IP_HITS.clear()


def test_access_password_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valida protecao por senha compartilhada (ACCESS_PASSWORD) e sessao por cookie."""
    from fastapi.testclient import TestClient

    from src.app import _IP_HITS, app

    _IP_HITS.clear()
    monkeypatch.setenv("ACCESS_PASSWORD", "uberhub-segredo")
    client = TestClient(app)

    # Healthcheck continua publico para o Render
    assert client.get("/health").status_code == 200

    # Pagina principal exibe tela de login e API bloqueia com 401
    login_html = client.get("/")
    assert login_html.status_code == 200
    assert "Acesso para Instrutores" in login_html.text
    assert (
        client.post("/api/predict-text", json={"statement": "Soma de dois inteiros."}).status_code
        == 401
    )

    # Senha errada retorna 401
    assert client.post("/api/login", json={"password": "errada"}).status_code == 401

    # Senha certa libera sessao
    ok_resp = client.post("/api/login", json={"password": "uberhub-segredo"})
    assert ok_resp.status_code == 200
    assert "uberhub_session" in client.cookies

    home = client.get("/")
    assert "Calibrador de Questões" in home.text
    assert (
        client.post("/api/predict-text", json={"statement": "Soma de dois inteiros."}).status_code
        == 200
    )

    # Logout encerra sessao
    assert client.post("/api/logout").status_code == 200
    assert (
        client.post("/api/predict-text", json={"statement": "Soma de dois inteiros."}).status_code
        == 401
    )
    _IP_HITS.clear()
