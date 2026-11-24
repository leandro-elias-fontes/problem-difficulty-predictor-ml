"""Carregamento de dados e gerador de dataset sintetico realista (300 problemas)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path("data/problems_sample.csv")
REAL_DATA_PATH = Path("data/real_problems.csv")


def rating_to_level(rating: float) -> str:
    """Converte o Rating numerico em uma faixa legivel para montagem de listas."""
    if rating < 1200:
        return "Iniciante"
    if rating < 1600:
        return "Intermediário"
    if rating < 2000:
        return "Avançado"
    if rating < 2400:
        return "Expert (ICPC Regional)"
    return "Grandmaster (ICPC Finals)"


def clean_text(text: object) -> str:
    """Remove tags HTML e normaliza espacos."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    no_html = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", no_html).strip()


def generate_problems_sample(
    output_path: Path = DEFAULT_DATA_PATH,
    n_samples: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera problemas sinteticos realistas (Rating 800-2800) cobrindo 7 faixas reais."""
    rng = np.random.default_rng(seed)

    archetypes = [
        {
            "rating_range": (800, 950),
            "n_choices": ["1 <= n <= 100", "1 <= a, b <= 1000", "1 <= n <= 10^5"],
            "time_choices": [1.0, 2.0],
            "templates": [
                "Leia dois inteiros a e b e imprima a soma simples dos dois números. {n_str}.",
                "Given two integers a and b, print their sum a + b. Constraints: {n_str}.",
                "Calcule a soma simples dos elementos de um array e imprima o total. {n_str}.",
                "Dado um inteiro n, verifique se ele é par ou ímpar. Limite: {n_str}.",
                "Leia três notas, calcule a média simples e imprima o maior valor. {n_str}.",
                "Dado um array de n inteiros, conte quantos elementos são pares. {n_str}.",
                "Verifique se a string informada é um palíndromo simples. Limite: {n_str}.",
                "Conte quantas vogais existem na palavra informada e imprima o resultado. {n_str}.",
                "Leia a idade de n alunos e imprima a maior e a menor idade. Limite: {n_str}.",
                "Calcule o fatorial de um número inteiro pequeno n. Limite: 1 <= n <= 12.",
            ],
        },
        {
            "rating_range": (950, 1200),
            "n_choices": ["1 <= n <= 1000", "1 <= n <= 10^5", "1 <= n <= 2 * 10^5"],
            "time_choices": [1.0, 2.0],
            "templates": [
                "Ordene o vetor de n inteiros e conte a frequência de cada valor no mapa. {n_str}.",
                "Use soma de prefixos e dois ponteiros para achar a janela deslizante. {n_str}.",
                "Simule a fila de atendimento e processe os eventos na ordem de chegada. {n_str}.",
                "Sort the array and use two pointers to find pairs with target sum. {n_str}.",
                (
                    "Use uma pilha (stack) para verificar se a sequência de parênteses "
                    "está balanceada. {n_str}."
                ),
                (
                    "Dado um vetor de preços, use janela deslizante para achar o maior "
                    "lucro. {n_str}."
                ),
                (
                    "Agrupe os alunos por pontuação usando dicionário/hash e ordene "
                    "pelo nome. {n_str}."
                ),
            ],
        },
        {
            "rating_range": (1200, 1500),
            "n_choices": ["1 <= n <= 10^5", "1 <= n <= 2 * 10^5"],
            "time_choices": [1.0, 2.0],
            "templates": [
                (
                    "Dados n intervalos disjuntos na reta, escolha o máximo de intervalos "
                    "de forma gulosa. Limite: {n_str}."
                ),
                (
                    "Encontre o menor valor x para dividir o vetor em k blocos usando "
                    "busca binária na resposta monótona. Limite: {n_str}."
                ),
                (
                    "Ordene a permutação minimizando o custo de trocas adjacentes com "
                    "estratégia gulosa. Limite: {n_str}."
                ),
                ("Calcule os números primos no intervalo usando crivo e mdc. Limite: {n_str}."),
                (
                    "Conte quantas formas existem de alocar tarefas usando busca binária "
                    "e fila de prioridade (heap). Limite: {n_str}."
                ),
                (
                    "Minimize the maximum distance between stations using binary search "
                    "on the answer and greedy check. Constraints: {n_str}."
                ),
            ],
        },
        {
            "rating_range": (1500, 1800),
            "n_choices": ["1 <= n, m <= 10^5", "2 <= n <= 2 * 10^5"],
            "time_choices": [2.0, 2.5],
            "templates": [
                (
                    "Dado um grafo conexo com n vértices e m arestas, encontre o menor "
                    "caminho entre 1 e n via BFS ou Dijkstra. Limite: {n_str}."
                ),
                (
                    "Verifique se o grafo direcionado possui ciclo e imprima uma ordenação "
                    "topológica válida usando DFS. Limite: {n_str}."
                ),
                (
                    "Conte subsequências válidas de tamanho k módulo 10^9 + 7 usando "
                    "programação dinâmica e knapsack (mochila). Limite: {n_str}."
                ),
                (
                    "Encontre a árvore geradora mínima usando Kruskal e union-find dsu "
                    "no grafo. Limite: {n_str}."
                ),
                (
                    "Dada uma malha de cidades e estradas com pesos, calcule o caminho "
                    "mínimo com Dijkstra e fila de prioridade. Limite: {n_str}."
                ),
                (
                    "Calcule o número de maneiras distintas de dar o troco para um valor "
                    "V usando programação dinâmica (DP) módulo 10^9 + 7. Limite: {n_str}."
                ),
            ],
        },
        {
            "rating_range": (1800, 2100),
            "n_choices": ["1 <= n <= 10^5", "1 <= n <= 2 * 10^5"],
            "time_choices": [2.0, 2.5],
            "templates": [
                (
                    "Dada uma árvore enraizada com n vértices, calcule o diâmetro e o "
                    "ancestral comum LCA para cada subárvore. Limite: {n_str}."
                ),
                (
                    "Calcule o conjunto independente máximo usando programação dinâmica "
                    "em árvore enraizada. Limite: {n_str}."
                ),
                ("Conte ocorrências de padrões usando trie e kmp sobre strings. Limite: {n_str}."),
                (
                    "Em uma árvore com n vértices, responda consultas de menor ancestral "
                    "comum (LCA) usando binary lifting. Limite: {n_str}."
                ),
                (
                    "Dado um grafo ponderado e programação dinâmica sobre subconjuntos, "
                    "calcule o custo mínimo de conexão. Limite: {n_str}."
                ),
            ],
        },
        {
            "rating_range": (2100, 2400),
            "n_choices": ["1 <= n, q <= 2 * 10^5", "1 <= n <= 2 * 10^5"],
            "time_choices": [2.0, 3.0],
            "templates": [
                (
                    "Responda q consultas e atualizações em intervalo usando árvore de "
                    "segmentos com lazy propagation. Limite: {n_str}."
                ),
                (
                    "Processe q consultas offline em uma matriz grid n por m usando "
                    "Fenwick tree e árvore de segmentos. Limite: {n_str}."
                ),
                (
                    "Dado um grafo com árvore enraizada, responda q consultas em "
                    "subárvore com programação dinâmica e lazy propagation. {n_str}."
                ),
                (
                    "Mantenha uma segment tree com lazy propagation para atualizar intervalos "
                    "e consultar o máximo em [l, r]. Limite: {n_str}."
                ),
            ],
        },
        {
            "rating_range": (2450, 2800),
            "n_choices": ["1 <= n <= 20", "1 <= n <= 10^9", "1 <= n <= 10^{18}"],
            "time_choices": [2.5, 3.0, 4.0],
            "templates": [
                (
                    "Conte caminhos hamiltonianos no caixeiro viajante em um grafo usando "
                    "bitmask programação dinâmica módulo 998244353. Limite: {n_str}."
                ),
                (
                    "Resolva a recorrência linear sobre uma matriz módulo 10^9 + 7 usando "
                    "exponenciação rápida de matrizes. Limite: {n_str}."
                ),
                (
                    "Otimize a programação dinâmica em matriz grid usando convex hull trick "
                    "e centroid decomposition módulo 998244353. Limite: {n_str}."
                ),
                (
                    "Calcule o fluxo máximo e corte mínimo com Dinic e heavy-light "
                    "decomposition HLD na árvore. Limite: {n_str}."
                ),
                (
                    "Use FFT/NTT para multiplicar polinômios em árvore com centroid "
                    "decomposition módulo 998244353. Limite: {n_str}."
                ),
            ],
        },
    ]

    rows: list[dict[str, object]] = []
    for idx in range(n_samples):
        arch = archetypes[idx % len(archetypes)]
        low, high = arch["rating_range"]
        raw_rating = float(rng.uniform(low, high))
        rating = float(np.clip(round(raw_rating / 50.0) * 50, 800, 2800))

        n_str = str(rng.choice(arch["n_choices"]))
        time_limit = float(rng.choice(arch["time_choices"]))
        template = str(rng.choice(arch["templates"]))

        rows.append(
            {
                "problem_id": f"P-{100 + idx}",
                "statement": template.replace("{n_str}", n_str),
                "time_limit": time_limit,
                "rating": rating,
            }
        )

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_dataset(
    csv_path: Path | None = None,
    generate_if_missing: bool = True,
    n_samples: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Carrega e limpa o dataset CSV de problemas (prioriza real_problems.csv se existir)."""
    if csv_path is not None:
        path = csv_path
    elif REAL_DATA_PATH.exists():
        path = REAL_DATA_PATH
    else:
        path = DEFAULT_DATA_PATH
    if not path.exists():
        if not generate_if_missing:
            raise FileNotFoundError(f"Dataset nao encontrado em {path}")
        generate_problems_sample(path, n_samples=n_samples, seed=seed)

    df = pd.read_csv(path)
    if "statement" not in df.columns:
        for alt in ("problem_statement", "description", "text"):
            if alt in df.columns:
                df["statement"] = df[alt]
                break
        else:
            df["statement"] = ""

    df["statement"] = df["statement"].map(clean_text)
    if "time_limit" not in df.columns:
        df["time_limit"] = 2.0
    else:
        df["time_limit"] = pd.to_numeric(df["time_limit"], errors="coerce").fillna(2.0)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    df["rating"] = df["rating"].clip(800, 3000)
    return df.reset_index(drop=True)
