# Calibrador de Questões — UberHub Code Club

Classificador profissional de dificuldade e tópicos para auxiliar instrutores e monitores na montagem de listas e provas (NEPS, Beecrowd, Codeforces e AtCoder).

## Recursos Principais

- **Modo Render Free (< 512 MB RAM)** + **IA Híbrida Gratuita**: usa `XGBoost + PyTorch MLP` treinados em **8.619 problemas reais** localmente, com suporte opcional a provedores gratuitos via variáveis de ambiente (`GROQ_API_KEY` ou `OPENROUTER_API_KEY`) e fallback automático para execução local (sem nunca expor chaves ao navegador).
- **4 Modos de Uso na Interface Web**:
  - **Texto**: enunciado + presets de limite `N`.
  - **Link**: consulta rating e tags oficiais direto nas APIs públicas do Codeforces e AtCoder, além de leitura de nível no Beecrowd e NEPS.
  - **Lista (Lote)**: analisa até 15 questões de uma vez, ordena de A até O e gera alertas pedagógicos de saltos bruscos de dificuldade.
  - **Print (`Ctrl+V`)**: OCR 100% offline via RapidOCR, com identificação automática de questões conhecidas pelo título/ID e edição de texto extraído.

## Deploy Gratuito no Render.com

O repositório já inclui `Dockerfile` enxuto (< 512 MB RAM) e `render.yaml` (`runtime: docker`, `plan: free`, `healthCheckPath: /health`). Basta conectar o repositório no Render como **Web Service** (ou **Blueprint**).

## Executar Localmente

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
uvicorn src.app:app --reload
```

Acesse `http://127.0.0.1:8000`.

## Deploy Gratuito (Hugging Face Spaces)

Veja o passo a passo completo em [DEPLOY_HF_SPACES.md](DEPLOY_HF_SPACES.md).

## Qualidade e Testes

```bash
ruff check .
ruff format --check .
pytest -v
```

