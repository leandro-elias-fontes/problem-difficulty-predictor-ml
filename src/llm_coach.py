"""Coach de IA hibrido: Groq / OpenRouter (env vars) com fallback para LLM local ou ML."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import httpx

LLM_REPO_ID = os.getenv("LOCAL_LLM_REPO", "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF")
LLM_FILENAME = os.getenv("LOCAL_LLM_FILE", "qwen2.5-coder-3b-instruct-q4_k_m.gguf")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def is_local_llm_enabled() -> bool:
    """Verifica se o Coach IA GGUF local esta habilitado (desligado em modo 512MB RAM)."""
    if os.getenv("LOW_MEMORY_MODE", "0").strip().lower() in {"1", "true", "yes"}:
        return False
    return os.getenv("ENABLE_LOCAL_LLM", "1").strip().lower() not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def _load_llm() -> object | None:
    """Baixa (se necessario) e carrega o modelo GGUF quantizado em memoria."""
    if not is_local_llm_enabled():
        return None
    try:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
    except ImportError:
        return None

    try:
        model_path = hf_hub_download(repo_id=LLM_REPO_ID, filename=LLM_FILENAME)
        threads = max(2, min(os.cpu_count() or 2, 4))
        return Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=threads,
            n_batch=256,
            verbose=False,
        )
    except Exception:
        return None


def _parse_llm_json(raw_text: str) -> dict[str, object] | None:
    """Extrai e valida o JSON estruturado retornado pelo LLM."""
    match = _JSON_BLOCK_RE.search(raw_text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "rating" not in data:
        return None
    try:
        rating = int(round(float(data["rating"]) / 50.0) * 50)
    except (TypeError, ValueError):
        return None
    rating = max(800, min(rating, 3000))

    raw_topics = data.get("topics") or []
    topics = (
        [str(t).strip() for t in raw_topics if str(t).strip()][:4]
        if isinstance(raw_topics, list)
        else []
    )
    return {
        "llm_rating": rating,
        "algorithm": str(data.get("algorithm") or "").strip()[:80],
        "complexity": str(data.get("complexity") or "").strip()[:50],
        "mentor_tip": str(data.get("mentor_tip") or "").strip()[:260],
        "topics": topics,
    }


def _build_coach_prompt(
    statement: str,
    n_display: str,
    time_limit: float,
    semantic_rating: int,
    semantic_topics: list[str],
) -> str:
    return (
        "Você é um coach especialista em Maratona de Programação (Codeforces/ICPC/OBI).\n"
        "Analise o problema abaixo e estime:\n"
        "1. rating (escala Codeforces de 800 a 2800, múltiplo de 50: "
        "800-1100 Iniciante, 1200-1500 Intermediário, 1600-1900 Avançado, 2000+ Expert)\n"
        "2. algorithm (técnica principal em português, ex: 'Dijkstra', 'Busca Binária')\n"
        "3. complexity (ex: 'O(N log N)', 'O(N + M)', 'O(2^N)')\n"
        "4. topics (lista com 1 a 3 tópicos em português)\n"
        "5. mentor_tip (1 frase curta para ajudar o monitor a guiar o aluno sem dar spoiler)\n\n"
        f"Enunciado: {statement[:1600]}\n"
        f"Limite detectado: {n_display} | Tempo: {time_limit}s\n"
        f"Referência estatística inicial: ~{semantic_rating} ({', '.join(semantic_topics)})\n\n"
        "Responda APENAS com um JSON válido no formato:\n"
        '{"rating": 1500, "algorithm": "...", "complexity": "...", '
        '"topics": ["..."], "mentor_tip": "..."}'
    )


def _call_openai_compatible_api(
    endpoint: str,
    api_key: str,
    model_name: str,
    prompt: str,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, object] | None:
    """Chama API compativel com OpenAI (Groq / OpenRouter) sem expor a chave."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Responda apenas em JSON válido, sem markdown ou texto extra.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 220,
    }
    try:
        resp = httpx.post(endpoint, headers=headers, json=payload, timeout=15.0)
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_llm_json(content)
    except Exception:
        return None


def analyze_with_local_llm(
    statement: str,
    n_display: str,
    time_limit: float,
    semantic_rating: int,
    semantic_topics: list[str],
) -> dict[str, object] | None:
    """Analisa com Groq/OpenRouter (se configurado via env) ou fallback para GGUF local."""
    prompt = _build_coach_prompt(statement, n_display, time_limit, semantic_rating, semantic_topics)

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        res = _call_openai_compatible_api(
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key=groq_key,
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            prompt=prompt,
        )
        if res is not None:
            return {**res, "provider": "Groq (Llama 3.3 70B)"}

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        res = _call_openai_compatible_api(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key=openrouter_key,
            model_name=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
            prompt=prompt,
            extra_headers={"X-Title": "Calibrador de Questoes"},
        )
        if res is not None:
            return {**res, "provider": "OpenRouter (Free)"}

    llm = _load_llm()
    if llm is None:
        return None

    try:
        out = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Responda apenas em JSON válido, sem markdown ou texto extra.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=220,
        )
        content = out["choices"][0]["message"]["content"]
        parsed = _parse_llm_json(content)
        if parsed is not None:
            return {**parsed, "provider": "Qwen2.5-Coder-3B (Local)"}
        return None
    except Exception:
        return None
