"""
Linhas de base. Este e o modulo mais importante do pacote.

A linha de base que mata a maioria das demonstracoes de IA em descoberta de
farmacos e a mais boba de todas, ranquear a biblioteca por similaridade de
Tanimoto aos ativos conhecidos. Se o seu pipeline nao bate isso, ele nao tem
valor, por mais sofisticado que seja.

Por isso essas funcoes vivem no pacote e nao num notebook, e por isso toda
etapa que produz ranking reporta o numero delas ao lado do seu.
"""
from __future__ import annotations

import random

from .chem import fingerprint, bulk_similarity
from .metrics import triagem


def rank_aleatorio(smiles_lib, seed=0):
    """Piso absoluto. Perder para isto significa que a pipeline e pior que nada."""
    idx = list(range(len(smiles_lib)))
    random.Random(seed).shuffle(idx)
    return idx


def rank_similaridade(smiles_lib, smiles_ativos, agregacao="max"):
    """
    O portao. Para cada composto da biblioteca, similaridade aos ativos
    conhecidos, agregada por maximo ou por media. Ordena do maior para o menor.

    Uma linha de conceito, notoriamente dificil de bater.
    """
    fps_ativos = [f for f in (fingerprint(s) for s in smiles_ativos) if f is not None]
    if not fps_ativos:
        raise ValueError("nenhum ativo valido")
    escores = []
    for s in smiles_lib:
        f = fingerprint(s)
        if f is None:
            escores.append(-1.0)
            continue
        sims = bulk_similarity(f, fps_ativos)
        escores.append(max(sims) if agregacao == "max" else sum(sims) / len(sims))
    return sorted(range(len(smiles_lib)), key=lambda i: -escores[i]), escores


def prever_media(y_treino, n):
    """Linha de base de regressao. Prever sempre a media do treino."""
    m = sum(y_treino) / len(y_treino)
    return [m] * n


def comparar(rotulos, rankings: dict):
    """
    Recebe os rotulos da biblioteca e um dicionario de nome para ordem de
    indices, e devolve a tabela de triagem de cada um.

    Use sempre com pelo menos aleatorio e similaridade dentro do dicionario.
    """
    saida = {}
    for nome, ordem in rankings.items():
        saida[nome] = triagem([rotulos[i] for i in ordem])
    return saida


def passou_no_portao(tabela, nome_modelo, nome_base="similaridade", metrica="ef_1pct"):
    """
    Decisao D4 em codigo. Devolve (passou, mensagem).

    Chame isto antes de avancar de etapa. Se devolver False, pare.
    """
    if nome_modelo not in tabela or nome_base not in tabela:
        return False, "tabela incompleta, rode as duas linhas de base"
    a = tabela[nome_modelo].get(metrica)
    b = tabela[nome_base].get(metrica)
    if a is None or b is None:
        return False, "metrica ausente"
    if a > b:
        return True, f"{nome_modelo} {a} supera {nome_base} {b} em {metrica}"
    return False, (
        f"{nome_modelo} {a} NAO supera {nome_base} {b} em {metrica}. "
        "Pela decisao D4 o projeto nao avanca daqui. Volte para curadoria e "
        "representacao antes de construir qualquer coisa em cima.")
