"""
Divisoes de treino e teste, e as assertivas de vazamento.

As assertivas nao sao prints, sao funcoes que levantam excecao. Elas entram
na suite de testes e quebram a construcao. Um repositorio que falha quando
alguem introduz vazamento e um repositorio em que da para confiar.
"""
from __future__ import annotations

import random

from .chem import fingerprint, bulk_similarity


class VazamentoError(AssertionError):
    """Levantada quando treino e teste compartilham algo que nao deveriam."""


def random_split(linhas, frac_teste=0.2, seed=42):
    idx = list(range(len(linhas)))
    random.Random(seed).shuffle(idx)
    corte = int(len(idx) * (1 - frac_teste))
    return [linhas[i] for i in idx[:corte]], [linhas[i] for i in idx[corte:]]


def scaffold_split(linhas, frac_teste=0.2, seed=42):
    """
    Separa arcaboucos inteiros. O analogo molecular do protein disjoint.
    Os maiores grupos vao para o treino, o que e a convencao da area e
    deixa o teste mais dificil de proposito.
    """
    grupos = {}
    for i, l in enumerate(linhas):
        grupos.setdefault(l.get("scaffold") or f"__sem__{i}", []).append(i)
    ordenados = sorted(grupos.values(), key=len, reverse=True)
    alvo_teste = int(len(linhas) * frac_teste)
    teste, treino = [], []
    for g in ordenados:
        if len(teste) + len(g) <= alvo_teste:
            teste.extend(g)
        else:
            treino.extend(g)
    return [linhas[i] for i in treino], [linhas[i] for i in teste]


def temporal_split(linhas, ano_corte):
    """
    Treina no que foi publicado ate o ano de corte, testa no que veio depois.
    E a divisao mais parecida com uso real e a mais dificil das tres.
    Linhas sem ano ficam no treino.
    """
    treino = [l for l in linhas if not l.get("year") or l["year"] <= ano_corte]
    teste = [l for l in linhas if l.get("year") and l["year"] > ano_corte]
    return treino, teste


# ---------------------------------------------------------------- assertivas

def assert_sem_composto_comum(treino, teste):
    a = {l["inchikey"] for l in treino}
    b = {l["inchikey"] for l in teste}
    comum = a & b
    if comum:
        raise VazamentoError(
            f"{len(comum)} compostos aparecem no treino e no teste, "
            f"exemplo {sorted(comum)[0]}")


def assert_sem_arcabouco_comum(treino, teste):
    a = {l["scaffold"] for l in treino if l.get("scaffold")}
    b = {l["scaffold"] for l in teste if l.get("scaffold")}
    comum = a & b
    if comum:
        raise VazamentoError(
            f"{len(comum)} arcaboucos aparecem nos dois lados, "
            f"exemplo {sorted(comum)[0]}")


def assert_sem_documento_comum(treino, teste):
    a = {d for l in treino for d in l.get("docs", [])}
    b = {d for l in teste for d in l.get("docs", [])}
    comum = a & b
    if comum:
        raise VazamentoError(
            f"{len(comum)} documentos de origem aparecem nos dois lados. "
            "No split temporal isso significa que o mesmo artigo vazou.")


def assert_sem_proteina_comum(treino, teste, col1="protein1", col2="protein2"):
    a = {l[col1] for l in treino} | {l[col2] for l in treino}
    b = {l[col1] for l in teste} | {l[col2] for l in teste}
    comum = a & b
    if comum:
        raise VazamentoError(f"{len(comum)} proteinas aparecem nos dois lados")


def similaridade_maxima(treino, teste, amostra=300, seed=0):
    """
    Maior similaridade de Tanimoto entre qualquer composto de teste e qualquer
    de treino. Nao levanta excecao, devolve o numero, que precisa ser
    registrado no relatorio. Valor perto de 1 com split por arcabouco indica
    que o arcabouco esta separando pouco.
    """
    rng = random.Random(seed)
    tr = [l["smiles"] for l in treino]
    te = [l["smiles"] for l in teste]
    if len(tr) > amostra:
        tr = rng.sample(tr, amostra)
    if len(te) > amostra:
        te = rng.sample(te, amostra)
    fps_tr = [f for f in (fingerprint(s) for s in tr) if f is not None]
    maior = 0.0
    for s in te:
        f = fingerprint(s)
        if f is None or not fps_tr:
            continue
        maior = max(maior, max(bulk_similarity(f, fps_tr)))
    return round(maior, 3)


def checar_tudo(treino, teste, modo):
    """
    Portao unico. Roda as assertivas que valem para o modo e devolve o
    relatorio. modo em aleatorio, arcabouco, temporal.
    """
    assert_sem_composto_comum(treino, teste)
    if modo == "arcabouco":
        assert_sem_arcabouco_comum(treino, teste)
    if modo == "temporal":
        assert_sem_documento_comum(treino, teste)
    return {
        "modo": modo,
        "n_treino": len(treino),
        "n_teste": len(teste),
        "similaridade_maxima": similaridade_maxima(treino, teste),
    }
