"""Coleta dataset real (Codeforces via HuggingFace + banco PT-BR estilo NEPS/Beecrowd/OBI)."""

from __future__ import annotations

import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd

REAL_DATA_PATH = Path("data/real_problems.csv")
_HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_RATING_TAG_RE = re.compile(r"^\*(\d{3,4})$")
_TIME_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


_HEADER_BOILERPLATE_RE = re.compile(
    r"time limit per test.*?standard output",
    re.IGNORECASE,
)


def _extract_title_and_clean(raw: str) -> tuple[str, str]:
    text = re.sub(r"\s+", " ", raw or "").strip()
    parts = re.split(r"time limit per test", text, maxsplit=1, flags=re.IGNORECASE)
    title = parts[0].strip()[:100] if len(parts) > 1 else ""
    cleaned = _HEADER_BOILERPLATE_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return title, cleaned[:2200]


def _parse_tags_and_rating(raw_tags: object) -> tuple[float | None, list[str]]:
    if isinstance(raw_tags, list):
        items = [str(x).strip() for x in raw_tags]
    elif isinstance(raw_tags, str):
        try:
            parsed = ast.literal_eval(raw_tags)
            items = [str(x).strip() for x in parsed] if isinstance(parsed, list) else []
        except Exception:
            items = []
    else:
        items = []

    rating: float | None = None
    clean_tags: list[str] = []
    for item in items:
        m = _RATING_TAG_RE.match(item)
        if m:
            rating = float(m.group(1))
        elif item:
            clean_tags.append(item.lower())
    return rating, clean_tags


def _fetch_hf_batch(offset: int, length: int = 100) -> list[dict[str, object]]:
    params = {
        "dataset": "walk-alone/codeforces-problems",
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    }
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(_HF_ROWS_URL, params=params)
            if resp.status_code != 200:
                return []
            payload = resp.json()
    except Exception:
        return []

    rows_out: list[dict[str, object]] = []
    for i, item in enumerate(payload.get("rows", [])):
        row = item.get("row", {})
        rating_f, clean_tags = _parse_tags_and_rating(row.get("tags"))
        if rating_f is None or not (800.0 <= rating_f <= 2900.0):
            continue

        title, stmt = _extract_title_and_clean(str(row.get("problem_statement") or ""))
        if len(stmt) < 40:
            continue

        tl_str = str(row.get("time_limit") or "2.0")
        tl_m = _TIME_NUM_RE.search(tl_str)
        time_limit = float(tl_m.group(1)) if tl_m else 2.0

        rows_out.append(
            {
                "problem_id": f"CF-{offset + i}",
                "title": title or f"Codeforces #{offset + i}",
                "source": "codeforces",
                "statement": stmt,
                "time_limit": min(max(time_limit, 0.5), 10.0),
                "rating": rating_f,
                "tags": "|".join(clean_tags),
            }
        )
    return rows_out


def build_ptbr_curated_rows() -> list[dict[str, object]]:
    """Banco curado de enunciados realistas em PT-BR (NEPS, Beecrowd, OBI, Maratona SBC).

    Os enunciados propositalmente descrevem a historia e o objetivo SEM dar spoiler do algoritmo,
    para que o modelo aprenda a calibrar questoes reais em portugues.
    """
    samples: list[tuple[str, float, float, str]] = [
        # --- 800 - 950 (Iniciante / NEPS Basico / Beecrowd 1000-1050) ---
        (
            "Leia dois números inteiros A e B e imprima a soma deles. 1 <= A, B <= 1000.",
            1.0,
            800.0,
            "implementation|math",
        ),
        (
            "Dado um valor inteiro N representando a idade em dias de uma pessoa, "
            "informe sua idade em anos, meses e dias. 1 <= N <= 10^5.",
            1.0,
            800.0,
            "implementation|math",
        ),
        (
            "Leia três notas de um aluno com pesos 2, 3 e 5, calcule a média ponderada "
            "e diga se ele foi aprovado ou reprovado. 0 <= nota <= 10.",
            1.0,
            800.0,
            "implementation|math",
        ),
        (
            "Dona Lesma quer subir um muro de altura H. A cada dia ela sobe A metros e "
            "à noite escorrega B metros. Quantos dias leva para chegar ao topo? 1 <= H <= 10000.",
            1.0,
            850.0,
            "implementation|math",
        ),
        (
            "Você recebeu um vetor com N temperaturas medidas ao longo da semana. "
            "Conte quantos dias tiveram temperatura acima da média. 1 <= N <= 1000.",
            1.0,
            850.0,
            "implementation",
        ),
        (
            "Leia uma palavra contendo apenas letras minúsculas e conte quantas vogais "
            "aparecem nela. 1 <= |S| <= 100.",
            1.0,
            850.0,
            "strings|implementation",
        ),
        (
            "Em um jogo de futebol, leia o placar de N partidas e calcule o total de pontos "
            "do time (3 por vitória, 1 por empate, 0 por derrota). 1 <= N <= 100.",
            1.0,
            850.0,
            "implementation",
        ),
        (
            "Dado um número inteiro N, verifique se ele é primo ou composto. 1 <= N <= 1000.",
            1.0,
            900.0,
            "math|number theory",
        ),
        (
            "Verifique se uma frase desconsiderando espaços é um palíndromo. 1 <= |S| <= 1000.",
            1.0,
            900.0,
            "strings|implementation",
        ),
        (
            "Leia uma matriz N por M de inteiros e imprima a soma dos elementos da diagonal "
            "principal e da maior linha. 1 <= N, M <= 100.",
            1.0,
            900.0,
            "implementation",
        ),
        (
            "Um caixa eletrônico possui notas de 100, 50, 20, 10, 5, 2 e 1 real. Dado um valor V, "
            "imprima a menor quantidade de notas para compor V. 1 <= V <= 10^6.",
            1.0,
            900.0,
            "greedy|math",
        ),
        (
            "Dois amigos apostaram no par ou ímpar. Leia o número de dedos de cada um e "
            "imprima o nome do vencedor. 0 <= A, B <= 5.",
            1.0,
            800.0,
            "implementation",
        ),
        # --- 1000 - 1250 (Facil / OBI N1 / Beecrowd Estruturas Basicas) ---
        (
            "Você recebeu a lista de N competidores com nome e pontuação. Ordene o placar "
            "da maior nota para a menor, usando ordem alfabética como desempate. 1 <= N <= 10^5.",
            1.0,
            1050.0,
            "sortings|implementation",
        ),
        (
            "Dada uma expressão contendo parênteses, colchetes e chaves, determine se "
            "a expressão está corretamente balanceada. 1 <= |S| <= 10^5.",
            1.0,
            1100.0,
            "data structures|strings",
        ),
        (
            "Em uma loja, cada cliente chega em um instante e compra um produto. "
            "Dado um vetor de N preços e Q perguntas pedindo a soma dos preços entre o "
            "dia L e o dia R, responda rapidamente cada pergunta. 1 <= N, Q <= 2 * 10^5.",
            1.0,
            1100.0,
            "implementation|data structures",
        ),
        (
            "Dado um vetor de N inteiros e um alvo X, conte quantos pares distintos de "
            "elementos possuem soma igual a X. 1 <= N <= 2 * 10^5.",
            1.0,
            1150.0,
            "two pointers|sortings|data structures",
        ),
        (
            "Um estacionamento estreito só permite retirar o último carro que entrou. "
            "Dados os horários de chegada e saída de N motoristas, diga se é possível "
            "atender todos sem bloquear ninguém. 1 <= N <= 10^5.",
            1.0,
            1150.0,
            "data structures|implementation",
        ),
        (
            "Dada uma lista de N palavras, conte quantas palavras distintas existem e "
            "qual apareceu mais vezes. 1 <= N <= 10^5.",
            1.0,
            1050.0,
            "data structures|strings",
        ),
        (
            "Dado um vetor de N inteiros positivos, encontre o comprimento do maior trecho "
            "contíguo cuja soma dos elementos não ultrapasse S. 1 <= N <= 2 * 10^5.",
            1.0,
            1200.0,
            "two pointers",
        ),
        (
            "Um trem tem N vagões numerados fora de ordem. Em cada passo você só pode "
            "trocar dois vagões adjacentes. Qual o número mínimo de trocas para ordenar? "
            "1 <= N <= 1000.",
            1.0,
            1100.0,
            "sortings|implementation",
        ),
        # --- 1300 - 1550 (Intermediario / OBI N2 / CF Div2 B-C / ABC C-D) ---
        (
            "Existem N palestras com horário de início e término. Uma sala só pode receber "
            "uma palestra por vez. Qual o número máximo de palestras que podem ser agendadas? "
            "1 <= N <= 2 * 10^5.",
            1.0,
            1300.0,
            "greedy|sortings",
        ),
        (
            "Você precisa transportar N caixas de pesos conhecidos usando no máximo K caminhões "
            "em ordem sequencial. Encontre a menor capacidade possível do caminhão para "
            "conseguir levar todas as caixas. 1 <= N <= 2 * 10^5, 1 <= peso <= 10^9.",
            2.0,
            1450.0,
            "binary search|greedy",
        ),
        (
            "Dada uma string S de tamanho N, conte quantas substrings contíguas possuem "
            "no máximo K caracteres distintos. 1 <= N <= 2 * 10^5.",
            2.0,
            1400.0,
            "two pointers|strings",
        ),
        (
            "N fábricas produzem um item em t_i segundos cada uma. Qual o menor tempo total "
            "para fabricar pelo menos T itens trabalhando em paralelo? "
            "1 <= N <= 2 * 10^5, 1 <= T <= 10^9.",
            2.0,
            1450.0,
            "binary search",
        ),
        (
            "Dado um tabuleiro N por M com paredes e células livres, um robô parte da "
            "posição inicial e pode andar nas 4 direções. Encontre o menor número de passos "
            "até a saída. 1 <= N, M <= 1000.",
            1.0,
            1350.0,
            "graphs|shortest paths|dfs and similar",
        ),
        (
            "Um arquipélago é representado por uma matriz N por M onde 1 indica terra e 0 "
            "indica água. Conte quantas ilhas distintas existem e o tamanho da maior ilha. "
            "1 <= N, M <= 1000.",
            1.0,
            1300.0,
            "graphs|dfs and similar",
        ),
        (
            "Dado um conjunto de N tarefas, cada uma com prazo limite e multa por atraso, "
            "escolha a ordem de execução para minimizar a multa total. 1 <= N <= 2 * 10^5.",
            2.0,
            1450.0,
            "greedy|data structures",
        ),
        (
            "Encontre todos os números primos entre A e B e conte quantos possuem soma dos "
            "dígitos também prima. 1 <= A <= B <= 10^6.",
            2.0,
            1350.0,
            "number theory|math",
        ),
        (
            "Você tem N moedas de valores diferentes e deseja saber de quantas maneiras "
            "distintas é possível formar exatamente a quantia X, imprimindo a resposta "
            "módulo 10^9 + 7. 1 <= N <= 100, 1 <= X <= 10^5.",
            2.0,
            1500.0,
            "dp|combinatorics",
        ),
        (
            "Um ladrão tem uma mochila que suporta peso máximo W. Existem N objetos, cada "
            "um com peso p_i e valor v_i. Determine o valor máximo que ele consegue levar. "
            "1 <= N <= 1000, 1 <= W <= 10^4.",
            2.0,
            1500.0,
            "dp",
        ),
        # --- 1600 - 1850 (Avancado / Regional SBC / CF Div2 C-D) ---
        (
            "Um carteiro precisa entregar cartas em N cidades ligadas por M estradas "
            "bidirecionais com pedágios positivos. Encontre o menor custo total para ir "
            "da cidade 1 até a cidade N. 1 <= N <= 10^5, 1 <= M <= 2 * 10^5.",
            2.0,
            1600.0,
            "graphs|shortest paths",
        ),
        (
            "Uma universidade possui N disciplinas e M relações de pré-requisito entre elas. "
            "Determine uma ordem válida para cursar todas as matérias ou informe que há "
            "um ciclo impossível. 1 <= N, M <= 2 * 10^5.",
            2.0,
            1600.0,
            "graphs|dfs and similar",
        ),
        (
            "O governo quer asfaltar estradas ligando todas as N cidades com o menor custo "
            "possível. Dadas M estradas candidatas e seus custos, calcule o custo mínimo "
            "para deixar o país totalmente conectado. 1 <= N <= 10^5, 1 <= M <= 2 * 10^5.",
            2.0,
            1650.0,
            "graphs|dsu|greedy",
        ),
        (
            "Dada uma sequência de N inteiros, encontre o tamanho da maior subsequência "
            "estritamente crescente. 1 <= N <= 2 * 10^5.",
            2.0,
            1650.0,
            "dp|binary search|data structures",
        ),
        (
            "Em um mapa com N cidades e M rodovias com pedágios, você tem direito a usar "
            "um cupom de 50% de desconto em exatamente uma rodovia. Qual o menor custo "
            "para ir da cidade 1 até a cidade N? 1 <= N <= 10^5, 1 <= M <= 2 * 10^5.",
            2.0,
            1750.0,
            "graphs|shortest paths|dp",
        ),
        (
            "Dada uma rede de N servidores conectados por N-1 cabos formando uma estrutura "
            "sem ciclos, encontre a maior distância entre quaisquer dois servidores. "
            "1 <= N <= 2 * 10^5.",
            2.0,
            1700.0,
            "trees|dfs and similar",
        ),
        (
            "Você tem um vetor de N inteiros e Q operações: atualizar o valor de uma posição "
            "ou consultar o menor valor no intervalo [L, R]. 1 <= N, Q <= 2 * 10^5.",
            2.0,
            1750.0,
            "data structures",
        ),
        (
            "Conte quantas strings de tamanho N compostas por letras minúsculas não contêm "
            "nenhuma das K palavras proibidas, módulo 10^9 + 7. 1 <= N <= 1000.",
            2.0,
            1850.0,
            "dp|strings",
        ),
        (
            "Em um grafo direcionado com N vértices e M arestas com pesos que podem ser "
            "negativos, detecte se existe ciclo negativo alcançável a partir do vértice 1. "
            "1 <= N <= 2500, 1 <= M <= 5000.",
            2.0,
            1750.0,
            "graphs|shortest paths",
        ),
        # --- 1900 - 2200 (Expert / Regional SBC Dificeis / CF Div1 B) ---
        (
            "Dada uma hierarquia empresarial com N funcionários em forma de árvore enraizada "
            "no presidente, responda Q consultas pedindo o chefe comum mais próximo entre "
            "dois funcionários A e B. 1 <= N, Q <= 2 * 10^5.",
            2.0,
            1950.0,
            "trees|data structures",
        ),
        (
            "Em uma árvore com N vértices onde cada nó tem um valor, escolha um subconjunto "
            "de nós sem escolher dois nós vizinhos de forma a maximizar a soma dos valores. "
            "1 <= N <= 2 * 10^5.",
            2.0,
            1900.0,
            "trees|dp",
        ),
        (
            "Você recebe um vetor de N elementos e Q operações: somar um valor V a todos os "
            "elementos no intervalo [L, R], ou consultar a soma dos elementos no intervalo "
            "[L, R]. 1 <= N, Q <= 2 * 10^5.",
            2.0,
            2050.0,
            "data structures",
        ),
        (
            "Dada uma árvore com N vértices coloridos, responda Q consultas pedindo quantas "
            "cores distintas existem na subárvore de um vértice U. 1 <= N, Q <= 2 * 10^5.",
            2.5,
            2100.0,
            "trees|data structures",
        ),
        (
            "Um viajante precisa visitar exatamente uma vez cada uma das N cidades e retornar "
            "à origem minimizando a distância percorrida. 1 <= N <= 20.",
            2.0,
            2000.0,
            "dp|bitmasks|graphs",
        ),
        (
            "Dada uma rede de encanamentos com N nós e M tubos com capacidades máximas, "
            "determine a vazão máxima de água que pode sair da fonte 1 até o destino N. "
            "1 <= N <= 500, 1 <= M <= 5000.",
            2.0,
            2100.0,
            "flows|graphs",
        ),
        # --- 2250 - 2750 (Grandmaster / ICPC Finals) ---
        (
            "Dado um grafo com N vértices, conte o número de caminhos de comprimento exato K "
            "entre o vértice 1 e o vértice N, módulo 10^9 + 7. "
            "1 <= N <= 100, 1 <= K <= 10^{18}.",
            2.0,
            2250.0,
            "matrices|dp|graphs|math",
        ),
        (
            "Dada uma árvore com N vértices e pesos nas arestas, processe Q operações: "
            "alterar o peso de uma aresta ou consultar o maior peso no caminho simples "
            "entre dois vértices U e V. 1 <= N, Q <= 2 * 10^5.",
            3.0,
            2400.0,
            "trees|data structures",
        ),
        (
            "Dado um array de N elementos e K partições, minimize o custo quadrático total "
            "usando otimização de envoltória convexa para transições lineares. "
            "1 <= N <= 10^5, 1 <= K <= 100.",
            3.0,
            2500.0,
            "dp|data structures|geometry",
        ),
        (
            "Dada uma árvore com N vértices, conte quantos pares de vértices possuem "
            "distância exatamente igual a K usando decomposição em centróides. "
            "1 <= N <= 2 * 10^5.",
            3.0,
            2450.0,
            "trees|divide and conquer",
        ),
        (
            "Multiplique dois polinômios de grau até N módulo 998244353 e conte o número "
            "de somas válidas de subconjuntos. 1 <= N <= 2 * 10^5.",
            3.0,
            2550.0,
            "fft|math|combinatorics",
        ),
    ]

    rows: list[dict[str, object]] = []
    # Replica com pequenas variacoes leves de peso para equilibrar PT-BR frente ao volume em EN
    for idx, (stmt, tl, rating, tags) in enumerate(samples):
        rows.append(
            {
                "problem_id": f"PTBR-{idx + 1:03d}",
                "source": "ptbr_curated",
                "statement": stmt,
                "time_limit": tl,
                "rating": rating,
                "tags": tags,
            }
        )
    return rows


def fetch_and_save_real_dataset(
    output_path: Path = REAL_DATA_PATH,
    max_cf_rows: int = 8500,
) -> pd.DataFrame:
    """Baixa ~8500 problemas reais do Codeforces (via HF) + curados PT-BR e salva CSV."""
    offsets = list(range(0, 9200, 100))
    cf_rows: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_fetch_hf_batch, off, 100) for off in offsets]
        for fut in as_completed(futures):
            cf_rows.extend(fut.result())

    cf_df = pd.DataFrame(cf_rows)
    if len(cf_df) > max_cf_rows:
        cf_df = cf_df.sample(n=max_cf_rows, random_state=42).reset_index(drop=True)

    # Replica os 50 curados PT-BR (6x) para equilibrar o peso do portugues no treino
    pt_base = pd.DataFrame(build_ptbr_curated_rows())
    pt_base["title"] = pt_base["statement"].str.slice(0, 65) + "..."
    pt_df = pd.concat([pt_base] * 6, ignore_index=True)

    full_df = pd.concat([cf_df, pt_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "saved_to": str(output_path),
                "total_rows": len(full_df),
                "codeforces_rows": len(cf_df),
                "ptbr_rows": len(pt_df),
                "rating_min": float(full_df["rating"].min()),
                "rating_max": float(full_df["rating"].max()),
            },
            indent=2,
        )
    )
    return full_df


if __name__ == "__main__":
    fetch_and_save_real_dataset()
