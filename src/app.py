"""Aplicacao web FastAPI (100% offline) para prever Rating por texto ou imagem."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.dataset import clean_text
from src.predict import predict_rating
from src.semantic_model import rating_to_uberhub_track
from src.url_fetcher import lookup_problem_by_url, match_known_problem_by_text
from src.vision_extract import extract_problems_from_image
from src.web_ui import HTML_PAGE

FEEDBACK_PATH = Path("data/feedback.jsonl")

app = FastAPI(
    title="Calibrador de Questoes - Maratona",
    description="Previsao de Rating (800-3000) de problemas de maratona com XGBoost + PyTorch.",
    version="0.3.0",
)


class TextPredictRequest(BaseModel):
    """Payload JSON para previsao via texto."""

    statement: str = Field(..., min_length=3)
    n_limit: str | None = None
    time_limit: float = Field(default=2.0, gt=0.0, le=30.0)


class UrlPredictRequest(BaseModel):
    """Payload JSON para previsao/consulta via link."""

    url: str = Field(..., min_length=8)


class FeedbackRequest(BaseModel):
    """Payload JSON para registrar feedback de calibracao dos instrutores."""

    rating: int = Field(..., ge=800, le=3500)
    verdict: str = Field(..., pattern="^(ok|too_high|too_low)$")
    title: str = ""
    statement_preview: str = ""


class BatchAnalyzeRequest(BaseModel):
    """Payload JSON para analisar uma lista de links ou enunciados de uma vez."""

    items: list[str] = Field(..., min_length=1, max_length=15)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve a interface web principal."""
    return HTML_PAGE


@app.get("/health")
def health() -> dict[str, str]:
    """Healthcheck simples para deploy."""
    return {"status": "ok"}


@app.post("/api/predict-text")
def api_predict_text(payload: TextPredictRequest) -> dict[str, object]:
    """Estima o Rating a partir de texto e limites (ou identifica se for questao oficial)."""
    cleaned = clean_text(payload.statement)
    if len(cleaned) < 3:
        raise HTTPException(
            status_code=400,
            detail="Informe um enunciado valido (pelo menos 3 caracteres).",
        )
    official = match_known_problem_by_text(cleaned)
    if official is not None:
        track_info = rating_to_uberhub_track(int(official["rating"]))
        return {
            **official,
            "track": track_info["track"],
            "slot": track_info["slot"],
            "guidance": track_info["guidance"],
            "similar_problems": [],
        }
    return predict_rating(
        statement=cleaned,
        n_limit=payload.n_limit,
        time_limit=payload.time_limit,
    )


@app.post("/api/predict-url")
def api_predict_url(payload: UrlPredictRequest) -> dict[str, object]:
    """Consulta rating oficial (Codeforces/AtCoder) ou estima a partir de link publico."""
    try:
        info = lookup_problem_by_url(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if info.get("mode") == "official":
        rating = int(info["rating"])
        track_info = rating_to_uberhub_track(rating)
        return {
            **info,
            "track": track_info["track"],
            "slot": track_info["slot"],
            "guidance": track_info["guidance"],
            "similar_problems": [],
        }

    pred = predict_rating(statement=str(info["statement"]))
    return {
        "title": info.get("title", "Questão via Link"),
        "source": info.get("source", "Web"),
        **pred,
    }


@app.post("/api/analyze-batch")
def api_analyze_batch(payload: BatchAnalyzeRequest) -> dict[str, object]:
    """Analisa uma lista de ate 15 links/enunciados, ordena por dificuldade e aponta saltos."""
    results: list[dict[str, object]] = []
    errors: list[str] = []

    for idx, raw_item in enumerate(payload.items, start=1):
        entry = raw_item.strip()
        if not entry:
            continue
        try:
            if entry.startswith(("http://", "https://")):
                info = lookup_problem_by_url(entry)
                if info.get("mode") == "official":
                    rating = int(info["rating"])
                    track_info = rating_to_uberhub_track(rating)
                    results.append(
                        {
                            **info,
                            "track": track_info["track"],
                            "slot": track_info["slot"],
                            "guidance": track_info["guidance"],
                            "similar_problems": [],
                        }
                    )
                else:
                    pred = predict_rating(statement=str(info["statement"]))
                    results.append({"title": info.get("title", f"Questão {idx}"), **pred})
            else:
                pred = predict_rating(statement=entry)
                short_title = entry[:65] + ("..." if len(entry) > 65 else "")
                results.append({"title": short_title, **pred})
        except Exception as exc:
            errors.append(f"Item {idx}: {exc}")

    if not results:
        raise HTTPException(
            status_code=400,
            detail="Nenhuma questao valida foi analisada. " + " | ".join(errors[:2]),
        )

    results.sort(key=lambda r: int(r["rating"]))
    alerts: list[str] = []
    for i in range(len(results) - 1):
        r1, r2 = int(results[i]["rating"]), int(results[i + 1]["rating"])
        if r2 - r1 >= 400:
            l1 = chr(65 + (i % 26))
            l2 = chr(65 + ((i + 1) % 26))
            mid = int(round(((r1 + r2) / 2) / 50.0) * 50)
            alerts.append(
                f"Salto brusco entre {l1} ({r1}) e {l2} ({r2}): "
                f"considere incluir uma questão intermediária (~{mid})."
            )

    if len(results) >= 3 and int(results[0]["rating"]) >= 1400:
        alerts.append(
            "A lista não tem questão de aquecimento (800–1100). "
            "Considere adicionar uma questão A mais acessível."
        )

    return {
        "count": len(results),
        "items": results,
        "alerts": alerts,
        "errors": errors,
    }


@app.post("/api/feedback")
def api_feedback(payload: FeedbackRequest) -> dict[str, str]:
    """Salva feedback rapido de instrutores/monitores para calibracao continua."""
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": payload.rating,
        "verdict": payload.verdict,
        "title": payload.title[:120],
        "statement_preview": payload.statement_preview[:300],
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "recorded"}


@app.post("/api/predict-image")
async def api_predict_image(
    file: UploadFile = File(...),  # noqa: B008
    n_limit: str | None = Form(default=None),  # noqa: B008
) -> dict[str, object]:
    """Extrai questoes de uma imagem via OCR local e estima o Rating de cada uma."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="O arquivo enviado esta vazio.")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Imagem muito grande (limite de 10 MB).")
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        problems = extract_problems_from_image(tmp_path)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    if not problems:
        raise HTTPException(
            status_code=400,
            detail="Nenhum texto legivel foi encontrado na imagem.",
        )

    results: list[dict[str, object]] = []
    for prob in problems:
        official = match_known_problem_by_text(prob.statement_text)
        if official is not None:
            track_info = rating_to_uberhub_track(int(official["rating"]))
            pred = {
                **official,
                "track": track_info["track"],
                "slot": track_info["slot"],
                "guidance": track_info["guidance"],
                "similar_problems": [],
            }
        else:
            pred = predict_rating(
                statement=prob.statement_text,
                n_limit=n_limit,
                time_limit=prob.time_limit,
            )
        results.append(
            {
                "label": prob.label,
                "time_limit": prob.time_limit,
                "ocr_text": prob.statement_text,
                **pred,
            }
        )
    return {"count": len(results), "results": results}
