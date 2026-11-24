"""Busca automatica por link (Codeforces API, AtCoder Kenkoooo API e paginas publicas)."""

from __future__ import annotations

import html
import re
from functools import lru_cache

import httpx

from src.dataset import rating_to_level

_CF_URL_RE = re.compile(
    r"codeforces\.com/(?:problemset/problem|contest|gym)/(\d+)/(?:problem/)?([A-Za-z0-9]+)",
    re.IGNORECASE,
)
_ATCODER_URL_RE = re.compile(
    r"atcoder\.jp/contests/([^/]+)/tasks/([A-Za-z0-9_]+)",
    re.IGNORECASE,
)

_CF_TAG_PT: dict[str, str] = {
    "dp": "Programação Dinâmica",
    "graphs": "Grafos",
    "trees": "Árvores",
    "greedy": "Guloso",
    "binary search": "Busca Binária",
    "data structures": "Estruturas de Dados",
    "math": "Matemática",
    "number theory": "Teoria dos Números",
    "implementation": "Implementação / Lógica",
    "strings": "Strings",
    "two pointers": "Dois Ponteiros",
    "sortings": "Ordenação",
    "shortest paths": "Menor Caminho",
    "dfs and similar": "DFS / BFS",
    "dsu": "Union-Find (DSU)",
    "combinatorics": "Combinatória",
    "bitmasks": "Bitmask",
    "flows": "Fluxo em Grafos",
    "geometry": "Geometria",
    "brute force": "Força Bruta",
    "constructive algorithms": "Algoritmos Construtivos",
    "divide and conquer": "Dividir e Conquistar",
    "fft": "FFT / Polinômios",
    "matrices": "Matrizes",
    "games": "Teoria dos Jogos",
    "probabilities": "Probabilidade",
    "hashing": "Hashing",
    "interactive": "Interativo",
}


def _translate_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        pt = _CF_TAG_PT.get(tag.lower())
        if pt and pt not in out:
            out.append(pt)
    return out[:5] if out else ["Lógica / Implementação"]


@lru_cache(maxsize=1)
def _load_cf_catalog() -> dict[tuple[int, str], dict[str, object]]:
    """Carrega e mantem em cache o catalogo oficial da API do Codeforces."""
    resp = httpx.get("https://codeforces.com/api/problemset.problems", timeout=15.0)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "OK":
        raise ValueError("API do Codeforces indisponivel no momento.")
    catalog: dict[tuple[int, str], dict[str, object]] = {}
    for prob in payload.get("result", {}).get("problems", []):
        cid = prob.get("contestId")
        idx = str(prob.get("index") or "").upper()
        if cid is not None and idx:
            catalog[(int(cid), idx)] = prob
    return catalog


@lru_cache(maxsize=1)
def _load_atcoder_catalog() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Carrega e mantem em cache o catalogo Kenkoooo do AtCoder."""
    with httpx.Client(timeout=15.0) as client:
        models_resp = client.get("https://kenkoooo.com/atcoder/resources/problem-models.json")
        probs_resp = client.get("https://kenkoooo.com/atcoder/resources/problems.json")
    models_resp.raise_for_status()
    probs_resp.raise_for_status()
    models = models_resp.json()
    probs = {str(p["id"]): p for p in probs_resp.json() if "id" in p}
    return models, probs


def _atcoder_diff_to_cf_scale(diff: float) -> int:
    """Converte a escala Kenkoooo do AtCoder para a escala equivalente (800-3000)."""
    # AtCoder diff <= 0 -> ~800; 400 -> ~1100; 800 -> ~1400; 1200 -> ~1700; 1600 -> ~2000
    mapped = 800.0 + max(diff, 0.0) * 0.75
    clamped = min(max(mapped, 800.0), 3000.0)
    return int(round(clamped / 50.0) * 50)


def _strip_html_to_text(raw_html: str) -> str:
    cleaned = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def lookup_problem_by_url(url: str) -> dict[str, object]:
    """Consulta metadados oficiais (Codeforces/AtCoder) ou extrai texto de link publico."""
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        raise ValueError("Informe uma URL valida comecando com https://")

    # 1. Codeforces (oficial via API)
    cf_match = _CF_URL_RE.search(clean_url)
    if cf_match:
        contest_id = int(cf_match.group(1))
        index = cf_match.group(2).upper()
        try:
            catalog = _load_cf_catalog()
        except Exception as exc:
            raise ValueError(
                "Nao foi possivel consultar a API do Codeforces agora. "
                "Cole o enunciado ou um print (Ctrl+V)."
            ) from exc
        prob = catalog.get((contest_id, index))
        if not prob or "rating" not in prob:
            raise ValueError(
                f"Questao {contest_id}{index} nao encontrada com rating oficial no Codeforces. "
                "Cole o enunciado ou um print (Ctrl+V) para estimar via ML."
            )
        rating = int(prob["rating"])
        tags = _translate_tags([str(t) for t in prob.get("tags", [])])
        return {
            "mode": "official",
            "source": "Codeforces",
            "title": f"CF {contest_id}{index} - {prob.get('name', '')}".strip(),
            "rating": rating,
            "level": rating_to_level(rating),
            "topics": tags,
            "confidence": "Oficial (Codeforces)",
            "n_display": "Rating oficial",
            "xgb_rating": float(rating),
            "mlp_rating": float(rating),
            "max_n_log10": 0.0,
        }

    # 2. AtCoder (oficial via Kenkoooo API)
    ac_match = _ATCODER_URL_RE.search(clean_url)
    if ac_match:
        task_id = ac_match.group(2)
        try:
            models, probs = _load_atcoder_catalog()
        except Exception as exc:
            raise ValueError(
                "Nao foi possivel consultar o catalogo do AtCoder agora. "
                "Cole o enunciado ou um print (Ctrl+V)."
            ) from exc
        model_info = models.get(task_id)
        if not model_info or model_info.get("difficulty") is None:
            raise ValueError(
                f"Dificuldade oficial nao disponivel para '{task_id}' no AtCoder. "
                "Cole o enunciado ou um print (Ctrl+V) para estimar via ML."
            )
        raw_diff = float(model_info["difficulty"])
        rating = _atcoder_diff_to_cf_scale(raw_diff)
        title = str(probs.get(task_id, {}).get("title") or task_id)
        return {
            "mode": "official",
            "source": "AtCoder",
            "title": f"AtCoder - {title}",
            "rating": rating,
            "level": rating_to_level(rating),
            "topics": [f"AtCoder Diff: {int(round(raw_diff))}"],
            "confidence": "Oficial (AtCoder)",
            "n_display": f"Diff {int(round(raw_diff))}",
            "xgb_rating": float(rating),
            "mlp_rating": float(rating),
            "max_n_log10": 0.0,
        }

    # 3. Outros sites publicos (NEPS, Beecrowd, etc.) -> tenta ler texto para rodar ML
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CalibradorQuestoes/1.0)"}
        resp = httpx.get(clean_url, headers=headers, timeout=12.0, follow_redirects=True)
        if resp.status_code >= 400:
            raise ValueError(
                "Este site bloqueou a leitura automatica do link. "
                "Tire um print da questao e pressione Ctrl+V na aba 'Print / Imagem', "
                "ou copie e cole o enunciado."
            )
        title_match = re.search(r"<title[^>]*>(.*?)</title>", resp.text, flags=re.S | re.I)
        page_title = (
            html.unescape(title_match.group(1)).strip() if title_match else "Questão via Link"
        )
        text = _strip_html_to_text(resp.text)
        if len(text) < 40:
            raise ValueError(
                "A pagina exige JavaScript ou login para exibir o enunciado. "
                "Use Ctrl+V com um print da tela ou cole o texto da questao."
            )
        return {
            "mode": "scraped_text",
            "source": "Web",
            "title": page_title[:90],
            "statement": text[:3000],
        }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Nao foi possivel acessar esse link agora. "
            "Copie o enunciado ou pressione Ctrl+V com um print da questao."
        ) from exc
