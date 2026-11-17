"""Aplicacao web FastAPI (100% offline) para prever Rating por texto ou imagem."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.predict import predict_rating
from src.vision_extract import extract_problems_from_image

app = FastAPI(
    title="Problem Difficulty Predictor",
    description="Previsao de Rating (800-3000) de problemas de maratona com XGBoost + PyTorch.",
    version="0.1.0",
)


class TextPredictRequest(BaseModel):
    """Payload JSON para previsao via texto."""

    statement: str = Field(..., min_length=3)
    n_limit: str | None = None
    time_limit: float = Field(default=2.0, gt=0.0, le=30.0)


_HTML_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Calibrador de Rating - Maratona de Programacao</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 2rem 1rem;
    }
    .container { max-width: 820px; margin: 0 auto; }
    h1 { font-size: 1.75rem; margin: 0 0 0.4rem; }
    .subtitle { color: var(--muted); margin: 0 0 1.5rem; font-size: 0.95rem; }
    .tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
    .tab-btn {
      background: var(--card);
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 0.6rem 1.1rem;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    .tab-btn.active { background: var(--accent); color: #0f172a; border-color: var(--accent); }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.4rem;
      margin-bottom: 1.25rem;
    }
    label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.35rem; }
    textarea, input[type="text"], input[type="number"] {
      width: 100%;
      background: #0f172a;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.7rem;
      font-size: 0.95rem;
      margin-bottom: 1rem;
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: 10px;
      padding: 2rem 1rem;
      text-align: center;
      cursor: pointer;
      margin-bottom: 1rem;
      color: var(--muted);
    }
    .dropzone:hover { border-color: var(--accent); color: var(--text); }
    .btn {
      background: var(--accent);
      color: #0f172a;
      border: none;
      padding: 0.75rem 1.4rem;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      width: 100%;
    }
    .btn:hover { background: var(--accent-hover); }
    .btn:disabled { opacity: 0.6; cursor: wait; }
    .hidden { display: none; }
    .result-item {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.2rem;
      margin-top: 0.9rem;
    }
    .rating-badge {
      display: inline-block;
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
      font-weight: 800;
      font-size: 1.1rem;
      margin-right: 0.6rem;
    }
    .meta { color: var(--muted); font-size: 0.88rem; margin-top: 0.5rem; }
    pre {
      background: #1e293b;
      padding: 0.7rem;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.82rem;
      color: #cbd5e1;
      white-space: pre-wrap;
    }
    .error { color: #f87171; margin-top: 0.75rem; font-size: 0.9rem; }
    #preview-img { max-width: 100%; max-height: 220px; margin-top: 0.7rem; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Calibrador de Rating - Maratona de Programacao</h1>
    <p class="subtitle">
      XGBoost + PyTorch MLP + OCR local (RapidOCR). 100% offline, sem APIs pagas.
    </p>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('text')">Texto e Limites</button>
      <button class="tab-btn" onclick="switchTab('image')">Print / Imagem (OCR)</button>
    </div>

    <div id="tab-text" class="card">
      <label for="statement">Enunciado ou resumo do problema</label>
      <textarea id="statement" rows="4"
        placeholder="Ex: Dada uma arvore com n vertices, responda q consultas modulo 10^9 + 7."
      ></textarea>
      <div class="row">
        <div>
          <label for="n-limit">Limite de N (opcional)</label>
          <input type="text" id="n-limit" placeholder="Ex: 2 * 10^5, 20, 10^{18}" />
        </div>
        <div>
          <label for="time-limit">Tempo limite (segundos)</label>
          <input type="number" id="time-limit" value="2.0" step="0.5" min="0.5" />
        </div>
      </div>
      <button class="btn" id="btn-text" onclick="predictText()">Estimar Rating</button>
    </div>

    <div id="tab-image" class="card hidden">
      <div class="dropzone" onclick="document.getElementById('img-input').click()">
        Clique para escolher um print (PNG, JPG, WEBP) da questao ou prova
        <input type="file" id="img-input" accept="image/*" class="hidden"
          onchange="previewFile(event)" />
        <div><img id="preview-img" class="hidden" alt="Preview" /></div>
      </div>
      <label for="img-n-limit">Forcar limite de N (opcional, util se o OCR errar expoente)</label>
      <input type="text" id="img-n-limit" placeholder="Ex: 10^5" />
      <button class="btn" id="btn-image" onclick="predictImage()">Ler Imagem e Estimar</button>
    </div>

    <div id="results"></div>
  </div>

  <script>
    function switchTab(tab) {
      document.querySelectorAll('.tab-btn').forEach((b, i) => {
        b.classList.toggle('active', (tab === 'text' && i === 0) || (tab === 'image' && i === 1));
      });
      document.getElementById('tab-text').classList.toggle('hidden', tab !== 'text');
      document.getElementById('tab-image').classList.toggle('hidden', tab !== 'image');
      document.getElementById('results').innerHTML = '';
    }

    function previewFile(event) {
      const file = event.target.files[0];
      const img = document.getElementById('preview-img');
      if (!file) { img.classList.add('hidden'); return; }
      img.src = URL.createObjectURL(file);
      img.classList.remove('hidden');
    }

    function ratingColor(rating) {
      if (rating < 1200) return '#94a3b8';
      if (rating < 1400) return '#4ade80';
      if (rating < 1600) return '#22d3ee';
      if (rating < 1900) return '#60a5fa';
      if (rating < 2100) return '#c084fc';
      if (rating < 2400) return '#fb923c';
      return '#f87171';
    }

    function renderItems(items) {
      const container = document.getElementById('results');
      container.innerHTML = items.map(item => `
        <div class="result-item">
          <div>
            <span class="rating-badge" style="background:${ratingColor(item.rating)};color:#0f172a">
              ${item.rating}
            </span>
            <strong>${item.label ? item.label + ' - ' : ''}${item.level}</strong>
          </div>
          <div class="meta">
            XGBoost: <b>${item.xgb_rating}</b> |
            PyTorch MLP: <b>${item.mlp_rating}</b> |
            log10(N): <b>${item.max_n_log10}</b>
            ${item.time_limit ? `| Tempo: <b>${item.time_limit}s</b>` : ''}
          </div>
          ${item.ocr_text ? `
            <details style="margin-top:0.6rem">
              <summary style="cursor:pointer;color:#94a3b8;font-size:0.85rem">
                Texto extraido pelo OCR
              </summary>
              <pre>${item.ocr_text}</pre>
            </details>` : ''}
        </div>
      `).join('');
    }

    async function predictText() {
      const btn = document.getElementById('btn-text');
      const statement = document.getElementById('statement').value.trim();
      if (!statement) return;
      btn.disabled = true;
      document.getElementById('results').innerHTML = '';
      try {
        const resp = await fetch('/api/predict-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            statement,
            n_limit: document.getElementById('n-limit').value.trim() || null,
            time_limit: parseFloat(document.getElementById('time-limit').value) || 2.0,
          }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Erro na previsao');
        renderItems([data]);
      } catch (err) {
        document.getElementById('results').innerHTML = `<div class="error">${err.message}</div>`;
      } finally {
        btn.disabled = false;
      }
    }

    async function predictImage() {
      const input = document.getElementById('img-input');
      if (!input.files[0]) return;
      const btn = document.getElementById('btn-image');
      btn.disabled = true;
      const resultsEl = document.getElementById('results');
      resultsEl.innerHTML = '<div class="meta">Lendo imagem via OCR local...</div>';
      try {
        const form = new FormData();
        form.append('file', input.files[0]);
        const nLim = document.getElementById('img-n-limit').value.trim();
        if (nLim) form.append('n_limit', nLim);
        const resp = await fetch('/api/predict-image', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Erro ao processar imagem');
        renderItems(data.results);
      } catch (err) {
        document.getElementById('results').innerHTML = `<div class="error">${err.message}</div>`;
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve a interface web principal."""
    return _HTML_PAGE


@app.get("/health")
def health() -> dict[str, str]:
    """Healthcheck simples para deploy."""
    return {"status": "ok"}


@app.post("/api/predict-text")
def api_predict_text(payload: TextPredictRequest) -> dict[str, object]:
    """Estima o Rating a partir de texto e limites."""
    return predict_rating(
        statement=payload.statement,
        n_limit=payload.n_limit,
        time_limit=payload.time_limit,
    )


@app.post("/api/predict-image")
async def api_predict_image(
    file: UploadFile = File(...),  # noqa: B008
    n_limit: str | None = Form(default=None),  # noqa: B008
) -> dict[str, object]:
    """Extrai questoes de uma imagem via OCR local e estima o Rating de cada uma."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    content = await file.read()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        problems = extract_problems_from_image(tmp_path)
    except ValueError as exc:
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
