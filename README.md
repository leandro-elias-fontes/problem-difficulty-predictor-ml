# Calibrador de Questões — Listas & Maratonas

Sistema completo em Python (FastAPI + Machine Learning + OCR offline) para estimar o **Rating** (escala 800–3000 estilo Codeforces), identificar **tópicos algorítmicos** e apoiar **instrutores e monitores** na montagem equilibrada de listas de treino e provas de programação competitiva (Codeforces, AtCoder, Beecrowd, NEPS Academy e OBI/SBC).

---

## Por que este projeto existe

Ao montar listas de exercícios ou simulados de maratona, ler apenas a história do enunciado engana: um mesmo problema de grafos ou contagem muda completamente de complexidade e nível conforme as restrições (`N <= 20` aponta para backtracking/bitmask DP; `N <= 10^5` aponta para Dijkstra/DSU/SegTree; `N <= 10^{18}` aponta para exponenciação de matrizes ou matemática).

Este projeto automatiza essa triagem combinando:
1. **Consulta oficial automática** para plataformas públicas (Codeforces API, AtCoder Kenkoooo, Beecrowd e NEPS).
2. **Pipeline de ML treinado em 8.619 problemas reais** (`XGBoost`, `PyTorch MLP`, `Ridge`, embeddings multilíngues PT-BR/EN e busca `k-NN` de questões parecidas).
3. **Cascata opcional de Coach IA via API gratuita** (`GROQ_API_KEY` → `OPENROUTER_API_KEY` → fallback 100% local), retornando técnica esperada, complexidade assintótica e dica pedagógica para monitoria.

---

## Principais Recursos

### 4 Modos de Análise na Interface Web
- **Texto:** cole o enunciado (em português ou inglês) com atalhos rápidos para limites de `N` (`≤ 20`, `10³`, `10⁵`, `2·10⁵`, `10¹⁸`) e tempo limite.
- **Link:** consulta rating e tags oficiais diretamente nas APIs públicas do Codeforces e AtCoder, além de leitura automática de nível no Beecrowd e NEPS Academy.
- **Lista (Lote):** analisa até **15 links ou enunciados** em sequência com progresso ao vivo, ordena automaticamente de `A` até `O`, gera resumo copiável e emite **alertas pedagógicos** (ex.: saltos bruscos de dificuldade ≥ 400 pontos ou ausência de questão de aquecimento).
- **Print (`Ctrl+V`):** arraste uma imagem ou cole um print direto da área de transferência. O **RapidOCR** extrai o texto 100% offline, identifica automaticamente se for uma questão conhecida pelo título/ID e permite editar o texto extraído e recalcular.

### Segurança & Hardening
- **Controle de acesso opcional por senha (`ACCESS_PASSWORD`):** exibe tela de login dedicada e protege todos os endpoints `/api/*` com cookie `HttpOnly` assinado via `HMAC-SHA256` (comparação em tempo constante) e proteção contra força bruta.
- **Proteção anti-SSRF:** bloqueia `localhost`, domínios `.local`/`.internal` e qualquer IP privado/loopback/link-local/reservado na leitura de URLs externas, validando redirecionamentos salto a salto.
- **Headers HTTP & Rate Limit:** aplica `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`, limite de requisições por IP (`/api/*`) e validação de tamanho máximo nos payloads.

---

## Estrutura do Projeto

- `src/app.py`: servidor FastAPI, middlewares de segurança/rate limit, autenticação por cookie assinado e rotas (`/`, `/health`, `/api/login`, `/api/logout`, `/api/predict-text`, `/api/predict-url`, `/api/predict-image`, `/api/analyze-batch`, `/api/feedback`).
- `src/web_ui.py`: interface web responsiva (desktop e mobile) servida diretamente pelo FastAPI.
- `src/features.py`: normalização Unicode PT-BR/EN, extração de `max_n_log10` (suporte a `10⁵`, `10^{18}`, `100.000`, ignorando módulos `10^9 + 7`), densidade matemática e 21 features estruturais/semânticas.
- `src/dataset.py` & `src/fetch_real_data.py`: coleta e unificação da base real (`data/real_problems.csv` com 8.619 problemas: 8.319 do Codeforces + 300 curados em PT-BR).
- `src/train.py`: treino e validação cruzada 5-Fold (`XGBRegressor` + `RatingMLP` em PyTorch) para o modo leve (`< 512 MB RAM`).
- `src/semantic_model.py`: pipeline semântico multilíngue (`paraphrase-multilingual-MiniLM-L12-v2`, classificador multi-label de 18 tópicos, ensemble `XGBoost + Ridge + PyTorch MLP` e índice `k-NN`).
- `src/llm_coach.py`: integração opcional com Groq / OpenRouter / GGUF local com fallback automático para o modelo local.
- `src/url_fetcher.py`: clientes com cache e proteção anti-SSRF para Codeforces, AtCoder, Beecrowd, NEPS e páginas públicas.
- `src/vision_extract.py`: OCR 100% offline com `rapidocr-onnxruntime` e separação automática de múltiplas questões.
- `src/predict.py`: orquestração de inferência (CLI e API web) com cache de artefatos.
- `tests/test_pipeline.py`: suíte `pytest` cobrindo regex, ausência de data leakage, PyTorch, OCR, endpoints FastAPI, anti-SSRF, rate limit e autenticação.

---

## Resultados Observados (5-Fold CV em 8.619 problemas reais)

| Modo / Modelo | MAE | RMSE | R² | Uso de RAM |
| --- | --- | --- | --- | --- |
| Modo Leve — `XGBoost` (`artifacts/`) | 355.4 | 441.7 | 0.475 | < 512 MB |
| Modo Leve — `PyTorch MLP` (`artifacts/`) | 375.2 | 486.4 | 0.364 | < 512 MB |
| Modo Semântico — Ensemble (`MiniLM + XGB + Ridge + MLP`) | 359.3 | 448.7 | 0.458 | ~900 MB |

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `ACCESS_PASSWORD` | *(vazio)* | Opcional: senha compartilhada para restringir o acesso a instrutores e monitores. |
| `PORT` | `7860` | Porta HTTP onde o servidor `uvicorn` escuta. |
| `LOW_MEMORY_MODE` | `0` | `1` usa apenas os modelos leves (`XGBoost + MLP`), cabendo com folga em 512 MB de RAM. |
| `ENABLE_LOCAL_LLM` | `1` | Habilita LLM GGUF local quando o extra `[local-llm]` está instalado. |
| `GROQ_API_KEY` | *(vazio)* | Opcional: ativa Coach IA via Groq (`llama-3.3-70b-versatile`) com fallback local. |
| `OPENROUTER_API_KEY` | *(vazio)* | Opcional: ativa fallback via OpenRouter (`llama-3.3-70b-instruct:free`). |
| `RATE_LIMIT_PER_MIN` | `60` | Máximo de requisições `/api/*` por IP por minuto (`0` desativa). |

---

## Como Executar

### 1. Localmente (Python)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
uvicorn src.app:app --reload
```

Acesse `http://127.0.0.1:8000`.

*(Opcional)* Para habilitar embeddings multilíngues (`MiniLM`) e LLM local (`llama-cpp-python`), instale o extra:
```bash
pip install -e ".[local-llm]"
```

### 2. Via CLI

```bash
python -m src.predict \
  --text "Dado um grafo conexo com n vértices e m arestas, encontre o menor caminho." \
  --n-limit "2 * 10^5" \
  --time-limit 2.0
```

Ou a partir de uma imagem (PNG/JPG/WEBP):
```bash
python -m src.predict --image caminho/da_questao.png
```

### 3. Via Docker

```bash
docker build -t calibrador-questoes .
docker run -p 7860:7860 -e ACCESS_PASSWORD=sua_senha calibrador-questoes
```

---

## Qualidade e Testes

```bash
ruff check .
ruff format --check .
pytest -v
```
