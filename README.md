# Previsor de Dificuldade de Problemas de Maratona (XGBoost + PyTorch)

Projeto em Python para estimar o Rating numérico (estilo Codeforces, de 800 a 2800+) de problemas de programação competitiva a partir do enunciado e das restrições matemáticas.

## Por que esse problema importa na prática

Quem monta listas de treino ou provas técnicas para maratonas de programação (ICPC/SBC, UberHub Code Club / UFU) precisa calibrar o nível de cada questão antes de publicar o contest. Ler apenas a história do enunciado engana: um mesmo problema de caminhos em grafos exige abordagens e níveis completamente diferentes dependendo dos limites (`N <= 20` aponta para bitmask DP; `N <= 10^5` aponta para BFS/Dijkstra; `N <= 10^{18}` aponta para exponenciação de matrizes).

Este repositório transforma essa leitura em um pipeline enxuto e sem data leakage, comparando um modelo baseado em árvores (`XGBRegressor`) e uma rede neural simples (`RatingMLP` em PyTorch) via validação cruzada `KFold`.

## Estrutura do Projeto

- `src/dataset.py`: carregamento e limpeza de CSVs de problemas + gerador offline em `data/problems_sample.csv` com 300 problemas sintéticos (Rating 800 a 2800).
- `src/features.py`: extração de features interpretáveis:
  - TF-IDF (unigramas e bigramas) sobre o enunciado.
  - Regex para `max_n_log10` (escala $\log_{10}$ do maior $N$, ignorando constantes de módulo como `10^9 + 7`), `time_limit_sec` e indicadores de estrutura (`has_graph`, `has_tree`, `has_matrix`, `has_modulo`).
- `src/train.py`: validação cruzada `KFold` (5 folds) comparando `XGBRegressor` e `RatingMLP` (`PyTorch`), calculando `MAE`, `RMSE` e `R2`, salvando os modelos em `artifacts/` e gerando dois gráficos em `assets/` (`loss_curve.png` e `pred_vs_actual.png`).
- `src/predict.py`: CLI rápida para estimar o Rating de um problema novo da aula (via texto ou imagem PNG/JPG/WEBP).
- `src/vision_extract.py`: OCR 100% offline via `rapidocr-onnxruntime` para ler prints/imagens (PNG/JPG/WEBP), separar questões e alimentar o `predict.py` sem nenhuma API externa.
- `src/app.py`: aplicação web FastAPI (interface HTML + endpoints `/api/predict-text` e `/api/predict-image`) rodando 100% local.
- `notebooks/kaggle_exploration.ipynb`: análise exploratória (EDA) e treino para portfólio/Kaggle.
- `tests/test_pipeline.py`: testes com `pytest` cobrindo regex, shapes/gradientes do PyTorch, smoke test de treino e ausência de data leakage.

## Como Executar

### 1. Instalar dependências

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### 2. Treinar e comparar os modelos (5-Fold CV)

```bash
python -m src.train
```

O comando gera `data/problems_sample.csv` (se ainda não existir), executa o `KFold` de 5 folds, salva os artefatos em `artifacts/` e gera `assets/loss_curve.png` e `assets/pred_vs_actual.png`.

### 3. Aplicação Web (FastAPI) ou CLI

Subir a aplicação web local (interface para texto e upload de imagem):

```bash
uvicorn src.app:app --reload
```

Acesse `http://127.0.0.1:8000` no navegador.

Ou use via CLI:

Por texto direto (100% offline):

```bash
python -m src.predict \
  --text "Dado um vetor de n inteiros, encontre o menor valor usando busca binaria." \
  --n-limit "2 * 10^5" \
  --time-limit 2.0
```

Por imagem (PNG/JPG/WEBP) da questão/prova (100% offline via OCR local):

```bash
python -m src.predict --image caminho/da_questao.png
```

### 4. Qualidade de Código e Testes

```bash
ruff check .
ruff format --check .
pytest -v
```

## Resultados Observados (KFold 5-Fold, `n=300`)

| Modelo | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| XGBoost Regressor | 66.65 | 89.36 | 0.978 |
| PyTorch MLP (`RatingMLP`) | 70.09 | 89.06 | 0.978 |

Os gráficos gerados em `assets/loss_curve.png` e `assets/pred_vs_actual.png` registram a convergência da MLP e a dispersão entre Rating real e predito.
