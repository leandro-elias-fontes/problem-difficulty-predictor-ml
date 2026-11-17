"""Engenharia de features simples e interpretaveis (TF-IDF + restricoes via regex)."""

from __future__ import annotations

import math
import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURE_COLS: list[str] = [
    "max_n_log10",
    "time_limit_sec",
    "has_queries",
    "has_basic",
    "has_ad_hoc_sort",
    "has_greedy_bs",
    "has_graph",
    "has_dp",
    "has_tree",
    "has_matrix",
    "has_modulo",
    "has_hard_tech",
    "complexity_score",
]

_SCI_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*(?:\*|x|\\cdot|·)\s*)?10\s*\^\s*\{?\s*(\d+)\s*\}?",
    re.IGNORECASE,
)
_E_RE = re.compile(r"\b(\d+(?:\.\d+)?)[eE]\+?(\d+)\b")
_BOUND_RE = re.compile(r"(?:<=|\\le|≤|<|=|ate|até)\s*(\d[\d,]*)\b", re.IGNORECASE)

_QUERIES_RE = re.compile(
    r"\b(consulta|consultas|query|queries|atualizacao|atualizacoes|update|updates|"
    r"intervalo \[l, r\]|q <=|n, q)\b",
    re.IGNORECASE,
)
_BASIC_RE = re.compile(
    r"\b(soma|somar|sum|adicao|subtracao|multiplicacao|dois inteiros|two integers|"
    r"dois numeros|dois valores|a \+ b|par ou impar|even or odd|pares|media|average|"
    r"tabuada|maior|menor|diferenca|produto|imprima|mostre|receba|leia|divisibilidade|"
    r"resto da divisao|palindromo|palindrome|vogais|fatorial|fibonacci|simples|basico|"
    r"iniciante|hello world|ola mundo)\b",
    re.IGNORECASE,
)
_ADHOC_SORT_RE = re.compile(
    r"\b(ordene|ordenar|ordenacao|sort|frequencia|frequency|mapa|hash|conjunto|"
    r"dois ponteiros|two pointers|janela deslizante|sliding window|soma de prefixos|"
    r"prefix sum|simulacao|fila)\b",
    re.IGNORECASE,
)
_GREEDY_BS_RE = re.compile(
    r"\b(guloso|gulosa|greedy|intervalos disjuntos|busca binaria|binary search|"
    r"monotona|monotonic|permutacao|trocas adjacentes|inversoes|crivo|primos|prime|"
    r"mdc|mmc|gcd|lcm|menor valor x|minimize the maximum|maximize the minimum)\b",
    re.IGNORECASE,
)
_GRAPH_RE = re.compile(
    r"\b(grafo|grafos|graph|graphs|vertice|vertices|vertex|aresta|arestas|edge|edges|"
    r"caminho|shortest path|bfs|dfs|dijkstra|bellman|floyd|ciclo|cycle|topologica|"
    r"topological|bipartido|bipartite|conexa|conexo|connected|kruskal|mst|dsu|"
    r"union-find|hamiltonian|hamiltonianos)\b",
    re.IGNORECASE,
)
_DP_RE = re.compile(
    r"\b(programacao dinamica|dynamic programming|\bdp\b|subsequencia|subsequencias|"
    r"subsequence|knapsack|mochila|troco|coin change|lcs|lis|recorrencia|transicao|"
    r"combinatoria|binomial|valor esperado|expected value)\b",
    re.IGNORECASE,
)
_TREE_RE = re.compile(
    r"\b(arvore|arvores|tree|trees|subarvore|subtree|enraizada|rooted|lca|"
    r"ancestral comum|diametro|segmentos|segment tree|segtree|lazy propagation|"
    r"fenwick|sparse table|trie|kmp|z-function)\b",
    re.IGNORECASE,
)
_MATRIX_RE = re.compile(
    r"\b(matriz|matrizes|matrix|matrices|grid|linha|coluna|row|column)\b",
    re.IGNORECASE,
)
_MODULO_RE = re.compile(
    r"(10\s*\^\s*\{?\s*9\s*\}?\s*\+\s*7|1000000007|998244353|\bmod\b|\bmodulo\b)",
    re.IGNORECASE,
)
_HARD_TECH_RE = re.compile(
    r"\b(bitmask|mascara de bits|hamiltoniano|hamiltonianos|caixeiro viajante|tsp|"
    r"convex hull trick|cht|li chao|centroid|centroide|heavy-light|hld|"
    r"exponenciacao rapida|matrix exponentiation|fft|ntt|automato|suffix array|"
    r"persistent|persistente|fluxo maximo|max flow|min cut|dinic|2-sat|grundy)\b",
    re.IGNORECASE,
)


def extract_max_n_log10(text: str, default_log10: float = 2.5) -> float:
    """Extrai log10 do maior limite de N no texto, ignorando constantes de modulo."""
    if not text:
        return default_log10

    cleaned = re.sub(r"10\s*\^\s*\{?\s*9\s*\}?\s*\+\s*7", " ", text)
    cleaned = re.sub(r"998244353|1000000007", " ", cleaned)
    candidates: list[float] = []

    for coeff_str, exp_str in _SCI_RE.findall(cleaned):
        exp = int(exp_str)
        if 1 <= exp <= 30:
            coeff = float(coeff_str) if coeff_str else 1.0
            candidates.append(math.log10(max(coeff, 1.0)) + exp)

    for coeff_str, exp_str in _E_RE.findall(cleaned):
        exp = int(exp_str)
        if 1 <= exp <= 30:
            coeff = float(coeff_str)
            candidates.append(math.log10(max(coeff, 1.0)) + exp)

    for num_str in _BOUND_RE.findall(cleaned):
        val = int(num_str.replace(",", ""))
        if 2 <= val <= 10**18:
            candidates.append(math.log10(val))

    if not candidates:
        return default_log10
    return float(min(max(candidates), 20.0))


def _compute_row_complexity(row: pd.Series) -> float:
    """Calcula um prior continuo de complexidade algoritmica (escala ~8.5 a ~26.5)."""
    if row["has_hard_tech"] > 0:
        base = 24.5
    elif row["has_tree"] > 0 and (row["has_dp"] > 0 or row["has_queries"] > 0):
        base = 20.5
    elif row["has_tree"] > 0 or (row["has_dp"] > 0 and row["has_graph"] > 0):
        base = 18.5
    elif row["has_dp"] > 0:
        base = 16.5
    elif row["has_graph"] > 0:
        base = 16.0
    elif row["has_greedy_bs"] > 0:
        base = 13.0
    elif row["has_ad_hoc_sort"] > 0:
        base = 10.5
    else:
        base = 8.5

    # Modificadores apenas quando ha estrutura nao-trivial (evita inflar soma simples com N=10^5)
    if base >= 10.5:
        if row["max_n_log10"] >= 5.0:
            base += 0.6
        if row["max_n_log10"] >= 12.0:
            base += 1.5
        if row["has_queries"] > 0:
            base += 0.8
        if row["has_modulo"] > 0:
            base += 0.7
        if row["has_matrix"] > 0 and row["has_hard_tech"] == 0:
            base += 0.5
    else:
        if row["max_n_log10"] <= 2.5:
            base = 8.2

    return float(base)


def extract_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai features numericas e flags de estrutura linha a linha (sem data leakage)."""
    texts = df["statement"].fillna("").astype(str)
    time_limits = (
        df["time_limit"].astype(float)
        if "time_limit" in df.columns
        else pd.Series(2.0, index=df.index)
    )

    feats = pd.DataFrame(index=df.index)
    feats["max_n_log10"] = texts.map(extract_max_n_log10)
    feats["time_limit_sec"] = time_limits.values
    feats["has_queries"] = texts.map(lambda t: float(bool(_QUERIES_RE.search(t))))
    feats["has_basic"] = texts.map(lambda t: float(bool(_BASIC_RE.search(t))))
    feats["has_ad_hoc_sort"] = texts.map(lambda t: float(bool(_ADHOC_SORT_RE.search(t))))
    feats["has_greedy_bs"] = texts.map(lambda t: float(bool(_GREEDY_BS_RE.search(t))))
    feats["has_graph"] = texts.map(lambda t: float(bool(_GRAPH_RE.search(t))))
    feats["has_dp"] = texts.map(lambda t: float(bool(_DP_RE.search(t))))
    feats["has_tree"] = texts.map(lambda t: float(bool(_TREE_RE.search(t))))
    feats["has_matrix"] = texts.map(lambda t: float(bool(_MATRIX_RE.search(t))))
    feats["has_modulo"] = texts.map(lambda t: float(bool(_MODULO_RE.search(t))))
    feats["has_hard_tech"] = texts.map(lambda t: float(bool(_HARD_TECH_RE.search(t))))

    adv_sum = (
        feats["has_ad_hoc_sort"]
        + feats["has_greedy_bs"]
        + feats["has_graph"]
        + feats["has_dp"]
        + feats["has_tree"]
        + feats["has_matrix"]
        + feats["has_modulo"]
        + feats["has_hard_tech"]
    )
    feats["has_basic"] = ((feats["has_basic"] > 0) | (adv_sum == 0)).astype(float)
    feats["complexity_score"] = feats.apply(_compute_row_complexity, axis=1)
    return feats


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Monta DataFrame com a coluna 'statement' e todas as NUMERIC_FEATURE_COLS."""
    num_df = extract_numeric_features(df)
    out = pd.DataFrame({"statement": df["statement"].fillna("").astype(str)}, index=df.index)
    for col in NUMERIC_FEATURE_COLS:
        out[col] = num_df[col].values
    return out


def build_preprocessor(max_tfidf_features: int = 150) -> ColumnTransformer:
    """Cria ColumnTransformer nao ajustado (TF-IDF + StandardScaler)."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=max_tfidf_features,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
                "statement",
            ),
            ("num", num_pipe, NUMERIC_FEATURE_COLS),
        ],
        remainder="drop",
    )
