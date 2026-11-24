"""Engenharia de features simples e interpretaveis (TF-IDF + restricoes via regex)."""

from __future__ import annotations

import math
import re
import unicodedata

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURE_COLS: list[str] = [
    "max_n_log10",
    "time_limit_sec",
    "num_constraints",
    "has_multi_vars",
    "statement_len_log",
    "math_density",
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
    "has_optimization",
    "has_counting",
    "is_small_n_exp",
    "is_large_n_nlogn",
    "is_huge_n_log",
]

_SCI_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*(?:\*|x|\\cdot)\s*)?10\s*\^\s*\{?\s*(\d+)\s*\}?",
    re.IGNORECASE,
)
_E_RE = re.compile(r"\b(\d+(?:\.\d+)?)[eE]\+?(\d+)\b")
_BOUND_RE = re.compile(r"(?:<=|\\le|<|=|ate)\s*(\d[\d.,]*)\b", re.IGNORECASE)
_MULTI_VAR_RE = re.compile(
    r"\b(n\s*,\s*m|n\s*,\s*q|n\s*,\s*k|n\s+vertices\s+e\s+m|n\s+cidades\s+e\s+m|"
    r"n\s+cities\s+and\s+m|n\s+por\s+m|n\s+by\s+m|n\s+x\s+m)\b",
    re.IGNORECASE,
)
_MATH_TOKEN_RE = re.compile(r"(<=|>=|\^|\bmod\b|%|\b[a-z]_[a-z0-9]\b|\[l,\s*r\]|\bxor\b|\bgcd\b)")

_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def normalize_for_matching(text: str) -> str:
    """Normaliza acentos, superscritos Unicode e simbolos matematicos para matching."""
    if not text:
        return ""
    out = re.sub(
        r"10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: f"10^{m.group(1).translate(_SUPERSCRIPT_MAP)}", text
    )
    out = out.replace("≤", "<=").replace("≥", ">=").replace("×", "*").replace("·", "*")
    decomposed = unicodedata.normalize("NFKD", out)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return ascii_text.lower()


_QUERIES_RE = re.compile(
    r"\b(consulta|consultas|query|queries|atualiza\w*|update|updates|"
    r"intervalo|range|q <=|n, q|operacoes de dois tipos|operations of two types|"
    r"perguntas|questions)\b",
    re.IGNORECASE,
)
_BASIC_RE = re.compile(
    r"\b(soma simples|dois inteiros|two integers|dois numeros|a \+ b|par ou impar|"
    r"even or odd|media ponderada|media simples|tabuada|vogais|fatorial|fibonacci|"
    r"hello world|ola mundo|idade em dias|notas de um aluno|placar de n partidas)\b",
    re.IGNORECASE,
)
_ADHOC_SORT_RE = re.compile(
    r"\b(ordene|ordenar|ordenacao|sort|sorted|frequencia|frequency|mapa|hash|dicionario|"
    r"conjunto|dois ponteiros|two pointers|janela deslizante|sliding window|"
    r"soma de prefixos|prefix sum|simulacao|fila|queue|pilha|stack|deque|"
    r"parenteses|brackets|subarray|trecho contiguo|substring|substrings|"
    r"distinct|distintos|distintas|pares|pairs|placar)\b",
    re.IGNORECASE,
)
_GREEDY_BS_RE = re.compile(
    r"\b(guloso|gulosa|greedy|intervalos|palestras|busca binaria|binary search|"
    r"monotona|monotonic|permutacao|permutation|trocas|swap|swaps|inversoes|inversions|"
    r"crivo|primos|prime|mdc|mmc|gcd|lcm|minimize the maximum|maximize the minimum|"
    r"menor capacidade|menor tempo|minimum time|minimum number of operations|"
    r"non-decreasing|nao decrescente|fila de prioridade|priority queue|heap|"
    r"prazo|deadline|multa|penalty|blocks|blocos)\b",
    re.IGNORECASE,
)
_GRAPH_RE = re.compile(
    r"\b(grafo|grafos|graph|graphs|vertice|vertices|vertex|aresta|arestas|edge|edges|"
    r"caminho minimo|menor caminho|menor custo|shortest path|minimum cost|cheapest|"
    r"bfs|dfs|busca em largura|busca em profundidade|dijkstra|bellman|floyd|"
    r"ciclo|cycle|topologica|topological|bipartido|bipartite|conexa|conexas|conexo|"
    r"connected|kruskal|mst|dsu|union-find|cidades|cities|estradas|roads|rodovias|"
    r"pontes|bridges|ilhas|islands|labirinto|maze|labyrinth|rede|network|"
    r"pre-requisito|prerequisite|bidirecionais|bidirectional|undirected|directed|"
    r"direcionado|pedagio|pedagios|alcancavel|reachable)\b",
    re.IGNORECASE,
)
_DP_RE = re.compile(
    r"\b(programacao dinamica|dynamic programming|\bdp\b|subsequencia|subsequencias|"
    r"subsequence|knapsack|mochila|troco|coin|coins|moedas|lcs|lis|recorrencia|"
    r"recurrence|transicao|combinatoria|combinatorics|binomial|valor esperado|"
    r"expected value|quantas formas|quantas maneiras|maneiras distintas|number of ways|"
    r"count the number|count how many|de quantas formas|subconjunto|subset|subsets|"
    r"backtracking|cupom|discount|particoes|partitions|crescente|increasing)\b",
    re.IGNORECASE,
)
_TREE_RE = re.compile(
    r"\b(arvore|arvores|tree|trees|subarvore|subtree|enraizada|rooted|lca|"
    r"ancestral comum|chefe comum|hierarquia|diametro|diameter|segmentos|"
    r"segment tree|segtree|lazy propagation|fenwick|sparse table|trie|kmp|z-function|"
    r"n-1|n - 1|sem ciclos|acyclic|leaf|leaves|folha|folhas|raiz|root)\b",
    re.IGNORECASE,
)
_MATRIX_RE = re.compile(
    r"\b(matriz|matrizes|matrix|matrices|grid|tabuleiro|board|linha|coluna|"
    r"row|column|celula|celulas|cell|cells)\b",
    re.IGNORECASE,
)
_MODULO_RE = re.compile(
    r"(10\s*\^\s*\{?\s*9\s*\}?\s*\+\s*7|1000000007|998244353|\bmod\b|\bmodulo\b)",
    re.IGNORECASE,
)
_HARD_TECH_RE = re.compile(
    r"\b(bitmask|mascara de bits|hamiltonian|hamiltonianos|caixeiro viajante|tsp|"
    r"visitar exatamente uma vez|visit every city|convex hull|cht|envoltoria convexa|"
    r"li chao|centroid|centroide|heavy-light|hld|exponenciacao rapida|"
    r"matrix exponentiation|fft|ntt|polinomios|polynomial|automato|suffix array|"
    r"suffix automaton|persistent|persistente|fluxo maximo|max flow|min cut|"
    r"vazao maxima|tubos com capacidades|dinic|2-sat|grundy|nim|xor basis)\b",
    re.IGNORECASE,
)
_OPT_RE = re.compile(
    r"\b(menor|maior|minimo|maximo|minimize|maximize|minimum|maximum|cheapest|"
    r"shortest|longest|optimal|otimo|otima)\b",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"\b(quantas|quantos|conte|count|number of ways|how many|maneiras|formas)\b",
    re.IGNORECASE,
)


def extract_max_n_log10(text: str, default_log10: float = 2.5) -> float:
    """Extrai log10 do maior limite de N no texto, ignorando constantes de modulo."""
    if not text:
        return default_log10

    normalized = normalize_for_matching(text)
    cleaned = re.sub(r"10\s*\^\s*\{?\s*9\s*\}?\s*\+\s*7", " ", normalized)
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
        # Trata separadores de milhar (ex: 100,000 ou 100.000) vs decimais curtos (ex: 2.5)
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", num_str):
            digits = num_str.replace(".", "")
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", num_str):
            digits = num_str.replace(",", "")
        else:
            digits = num_str.split(".")[0].replace(",", "")
        if digits.isdigit():
            val = int(digits)
            if 2 <= val <= 10**18:
                candidates.append(math.log10(val))

    if not candidates:
        return default_log10
    return float(min(max(candidates), 20.0))


def extract_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai features numericas, estruturais e semanticas bilingues (sem data leakage)."""
    raw_texts = df["statement"].fillna("").astype(str)
    texts = raw_texts.map(normalize_for_matching)
    time_limits = (
        df["time_limit"].astype(float)
        if "time_limit" in df.columns
        else pd.Series(2.0, index=df.index)
    )

    feats = pd.DataFrame(index=df.index)
    feats["max_n_log10"] = raw_texts.map(extract_max_n_log10)
    feats["time_limit_sec"] = time_limits.values
    feats["num_constraints"] = texts.map(lambda t: float(min(len(_BOUND_RE.findall(t)), 20)))
    feats["has_multi_vars"] = texts.map(lambda t: float(bool(_MULTI_VAR_RE.search(t))))
    feats["statement_len_log"] = raw_texts.map(lambda t: float(math.log1p(len(t))))
    feats["math_density"] = texts.map(
        lambda t: float(min(len(_MATH_TOKEN_RE.findall(t)), 30)) / max(len(t) / 100.0, 1.0)
    )

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
    feats["has_optimization"] = texts.map(lambda t: float(bool(_OPT_RE.search(t))))
    feats["has_counting"] = texts.map(lambda t: float(bool(_COUNT_RE.search(t))))

    feats["is_small_n_exp"] = (
        (feats["max_n_log10"] <= 1.45)
        & ((feats["has_counting"] > 0) | (feats["has_dp"] > 0) | (feats["has_graph"] > 0))
    ).astype(float)
    feats["is_large_n_nlogn"] = (
        (feats["max_n_log10"] >= 4.5) & (feats["max_n_log10"] <= 6.5)
    ).astype(float)
    feats["is_huge_n_log"] = (feats["max_n_log10"] >= 8.5).astype(float)
    return feats


_TOPIC_LABELS: list[tuple[str, str]] = [
    ("has_hard_tech", "Técnica Avançada"),
    ("has_tree", "Árvores / SegTree"),
    ("has_dp", "Programação Dinâmica"),
    ("has_graph", "Grafos"),
    ("has_greedy_bs", "Guloso / Busca Binária"),
    ("has_ad_hoc_sort", "Ordenação / Dois Ponteiros"),
    ("has_queries", "Consultas em Intervalo"),
    ("has_matrix", "Matriz / Grid"),
    ("has_modulo", "Aritmética Modular"),
]


def extract_detected_topics(feats_row: pd.Series) -> list[str]:
    """Retorna lista legivel de topicos detectados para exibir a instrutores."""
    topics = [label for col, label in _TOPIC_LABELS if float(feats_row.get(col, 0.0)) > 0.0]
    if not topics:
        topics.append("Lógica / Básico")
    return topics


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
                    strip_accents="unicode",
                ),
                "statement",
            ),
            ("num", num_pipe, NUMERIC_FEATURE_COLS),
        ],
        remainder="drop",
    )
