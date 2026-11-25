FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOW_MEMORY_MODE=1 \
    ENABLE_LOCAL_LLM=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY data/ ./data/

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" \
    && useradd -m -u 1000 user \
    && chown -R user:user /app

USER user

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
