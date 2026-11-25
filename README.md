# Calibrador de Questões — UberHub Code Club

Classificador profissional de dificuldade e tópicos para auxiliar instrutores e monitores na montagem de listas e provas (NEPS, Beecrowd, Codeforces e AtCoder).

## Recursos Principais

- **Modo Render Free (< 512 MB RAM)** + **IA Híbrida Gratuita**: usa `XGBoost + PyTorch MLP` treinados em **8.619 problemas reais** localmente, com cascata gratuita via variáveis de ambiente (`GROQ_API_KEY` → `OPENROUTER_API_KEY` → fallback local) sem expor chaves ao navegador ou a subprocessos.
- **4 Modos de Uso na Interface Web**:
  - **Texto**: enunciado + presets de limite `N`.
  - **Link**: consulta rating e tags oficiais direto nas APIs públicas do Codeforces e AtCoder, além de leitura de nível no Beecrowd e NEPS (com proteção anti-SSRF).
  - **Lista (Lote)**: analisa até 15 questões em sequência (sem timeout), ordena de A até O e gera alertas pedagógicos de saltos bruscos de dificuldade.
  - **Print (`Ctrl+V`)**: OCR 100% offline via RapidOCR, com identificação automática de questões conhecidas pelo título/ID e edição de texto extraído.
- **Segurança & Hardening**: headers HTTP de segurança (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`), rate limit por IP (`/api/*`), proteção anti-SSRF e limites máximos de payload.

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `ACCESS_PASSWORD` | *(vazio)* | Opcional: senha compartilhada para restringir o acesso a instrutores e monitores (exibe tela de login). |
| `PORT` | `7860` | Porta HTTP injetada pelo Render/Docker. |
| `LOW_MEMORY_MODE` | `0` (`1` no Render) | Usa apenas modelos leves (< 512 MB RAM), sem carregar `MiniLM`. |
| `ENABLE_LOCAL_LLM` | `1` (`0` no Render) | Habilita/desabilita LLM GGUF local quando `local-llm` está instalado. |
| `GROQ_API_KEY` | *(vazio)* | Opcional (gratuita): habilita análise via Groq (`Llama 3.3 70B`). |
| `OPENROUTER_API_KEY` | *(vazio)* | Opcional (gratuita): fallback após Groq via OpenRouter Free. |
| `RATE_LIMIT_PER_MIN` | `60` | Máximo de requisições `/api/*` por IP por minuto (`0` desativa). |

## Executar Localmente

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
uvicorn src.app:app --reload
```

Acesse `http://127.0.0.1:8000`.

## Deploy Gratuito no Render.com (Docker Free — 512 MB RAM)

1. Acesse [dashboard.render.com](https://dashboard.render.com) e clique em **New +** → **Blueprint** (ou **Web Service**).
2. Conecte este repositório (`leandro-elias-fontes/problem-difficulty-predictor-ml`).
3. O [render.yaml](render.yaml) já configura `runtime: docker`, `plan: free`, `healthCheckPath: /health`, `LOW_MEMORY_MODE=1` e `ENABLE_LOCAL_LLM=0`.
4. *(Opcional)* Adicione `GROQ_API_KEY` ou `OPENROUTER_API_KEY` nas **Environment Variables** do Render para raciocínio gratuito em nuvem, ou deixe vazio para rodar 100% local.
5. Clique em **Deploy** e acesse a URL pública gerada pelo Render.

## Qualidade e Testes

```bash
ruff check .
ruff format --check .
pytest -v
```

