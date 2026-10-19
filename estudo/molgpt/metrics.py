"""
Metricas. Regressao, triagem e calibracao.

A metrica que decide o projeto nao e erro medio, e enriquecimento no topo,
porque voce so vai conseguir testar algumas dezenas de compostos.
"""
from __future__ import annotations

import math


def _rank(xs):
    ordem = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(ordem):
        j = i
        while j + 1 < len(ordem) and xs[ordem[j + 1]] == xs[ordem[i]]:
            j += 1
        media = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[ordem[k]] = media
        i = j + 1
    return r


def pearson(y, p):
    n = len(y)
    if n < 2:
        return float("nan")
    my, mp = sum(y) / n, sum(p) / n
    num = sum((a - my) * (b - mp) for a, b in zip(y, p))
    dy = math.sqrt(sum((a - my) ** 2 for a in y))
    dp = math.sqrt(sum((b - mp) ** 2 for b in p))
    return num / (dy * dp) if dy and dp else float("nan")


def spearman(y, p):
    return pearson(_rank(y), _rank(p))


def concordance_index(y, p):
    """
    Fracao dos pares concordantes. Metrica padrao em afinidade, porque mede
    ordenacao e nao magnitude. Empate no predito conta meio ponto.
    """
    bons = total = 0.0
    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                continue
            total += 1
            maior_y, menor_y = (i, j) if y[i] > y[j] else (j, i)
            if p[maior_y] > p[menor_y]:
                bons += 1
            elif p[maior_y] == p[menor_y]:
                bons += 0.5
    return bons / total if total else float("nan")


def regressao(y, p):
    n = len(y)
    mse = sum((a - b) ** 2 for a, b in zip(y, p)) / n
    mae = sum(abs(a - b) for a, b in zip(y, p)) / n
    return {
        "n": n,
        "mse": round(mse, 4),
        "rmse": round(math.sqrt(mse), 4),
        "mae": round(mae, 4),
        "pearson": round(pearson(y, p), 4),
        "spearman": round(spearman(y, p), 4),
        "ci": round(concordance_index(y, p), 4),
    }


# ------------------------------------------------------------------ triagem

def enrichment_factor(rotulos_ordenados, frac):
    """
    Fator de enriquecimento. `rotulos_ordenados` e a lista de 0 e 1 na ordem
    do ranking do modelo, do melhor para o pior.

    EF igual a 1 significa nenhum ganho sobre o acaso.
    """
    n = len(rotulos_ordenados)
    k = max(1, int(round(n * frac)))
    ativos_total = sum(rotulos_ordenados)
    if ativos_total == 0:
        return float("nan")
    ativos_topo = sum(rotulos_ordenados[:k])
    return round((ativos_topo / k) / (ativos_total / n), 3)


def hit_rate_at(rotulos_ordenados, k):
    k = min(k, len(rotulos_ordenados))
    if k == 0:
        return float("nan")
    return round(sum(rotulos_ordenados[:k]) / k, 4)


def triagem(rotulos_ordenados):
    return {
        "n": len(rotulos_ordenados),
        "ativos": sum(rotulos_ordenados),
        "ef_1pct": enrichment_factor(rotulos_ordenados, 0.01),
        "ef_0_1pct": enrichment_factor(rotulos_ordenados, 0.001),
        "hit_rate_50": hit_rate_at(rotulos_ordenados, 50),
        "hit_rate_100": hit_rate_at(rotulos_ordenados, 100),
    }


# --------------------------------------------------------------- calibracao

def calibracao(erros, incertezas, n_faixas=5):
    """
    Agrupa as predicoes por faixa de incerteza e devolve o erro medio em cada
    uma. Se o erro nao cresce com a incerteza, a incerteza nao serve para
    filtrar, e filtrar por ela e o unico uso que ela tem.
    """
    pares = sorted(zip(incertezas, erros))
    n = len(pares)
    if n < n_faixas:
        return []
    tam = n // n_faixas
    faixas = []
    for i in range(n_faixas):
        ini = i * tam
        fim = n if i == n_faixas - 1 else (i + 1) * tam
        bloco = pares[ini:fim]
        faixas.append({
            "faixa": i + 1,
            "n": len(bloco),
            "incerteza_media": round(sum(u for u, _ in bloco) / len(bloco), 4),
            "erro_medio": round(sum(e for _, e in bloco) / len(bloco), 4),
        })
    return faixas


def incerteza_util(faixas):
    """True quando o erro medio cresce monotonicamente com a incerteza."""
    e = [f["erro_medio"] for f in faixas]
    return all(e[i] <= e[i + 1] for i in range(len(e) - 1))


# ------------------------------------------------------------------ amostral

def erro_padrao_proporcao(p, n):
    """Erro padrao de uma proporcao. Para reportar validade como 56 mais ou menos X."""
    return round(math.sqrt(max(p * (1 - p), 0.0) / max(n, 1)), 4)
