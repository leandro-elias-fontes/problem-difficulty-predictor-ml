"""Extracao offline de texto de imagens (PNG/JPG/WEBP) via OCR local (RapidOCR)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_QUESTION_HEADER_RE = re.compile(
    r"(?:^|\n)\s*((?:Problem|Questao|Questão|Problema)\s+[A-Z0-9]+\b[^\n]*)",
    re.IGNORECASE,
)
_TIME_LIMIT_RE = re.compile(
    r"(?:time\s*limit(?:\s*per\s*test)?|tempo\s*limite)\s*:?\s*(\d+(?:\.\d+)?)\s*(s|sec|seconds|ms)?",
    re.IGNORECASE,
)


@dataclass
class ExtractedProblem:
    """Representa uma questao extraida localmente de uma imagem via OCR."""

    label: str
    statement_text: str
    time_limit: float = 2.0


@lru_cache(maxsize=1)
def _get_ocr_engine() -> object:
    """Inicializa e reutiliza o motor RapidOCR em memoria."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _validate_image_path(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")
    if image_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato nao suportado: {image_path.suffix}. Use PNG, JPG, WEBP ou BMP.")


def extract_text_from_image(image_path: Path) -> str:
    """Executa OCR local 100% offline usando RapidOCR (ONNX Runtime)."""
    _validate_image_path(image_path)

    engine = _get_ocr_engine()
    try:
        result, _ = engine(str(image_path))
    except Exception as exc:
        raise ValueError(
            "Nao foi possivel ler a imagem. Verifique se o arquivo PNG/JPG/WEBP e valido."
        ) from exc

    if not result:
        return ""
    lines = [str(item[1]).strip() for item in result if len(item) >= 2 and str(item[1]).strip()]
    joined = "\n".join(lines)
    return re.sub(r"10\s*\^\s*(\d+)", r"10^\1", joined)


def _extract_time_limit(text: str, default: float = 2.0) -> float:
    match = _TIME_LIMIT_RE.search(text)
    if not match:
        return default
    val = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit == "ms":
        val /= 1000.0
    return val


def split_problems_from_ocr(raw_text: str) -> list[ExtractedProblem]:
    """Separa o texto OCR em questoes quando houver cabecalhos (ex: Problem A, Questao 1)."""
    cleaned = raw_text.strip()
    if not cleaned:
        return []

    matches = list(_QUESTION_HEADER_RE.finditer(cleaned))
    if len(matches) <= 1:
        label = matches[0].group(1).strip() if matches else "Questao"
        return [
            ExtractedProblem(
                label=label,
                statement_text=cleaned,
                time_limit=_extract_time_limit(cleaned),
            )
        ]

    problems: list[ExtractedProblem] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        chunk = cleaned[start:end].strip()
        problems.append(
            ExtractedProblem(
                label=match.group(1).strip(),
                statement_text=chunk,
                time_limit=_extract_time_limit(chunk),
            )
        )
    return problems


def extract_problems_from_image(image_path: Path) -> list[ExtractedProblem]:
    """Le uma imagem e retorna a lista de questoes extraidas offline."""
    _validate_image_path(image_path)
    raw_text = extract_text_from_image(image_path)
    return split_problems_from_ocr(raw_text)
