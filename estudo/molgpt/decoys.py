"""
Decoys pareados por propriedade.

Decoy sorteado ao acaso e facil demais e infla o enriquecimento. O modelo
aprende a reconhecer molecula do tamanho certo e o numero fica falso.

Aqui cada ativo recebe decoys com massa, logP, carga e contagem de doadores
e aceptores parecidos, mas topologia diferente, medida por Tanimoto baixo.
"""
from __future__ import annotations

import random

from .chem import properties, fingerprint, bulk_similarity

TOLERANCIA = {"mw": 25.0, "logp": 1.0, "hbd": 1, "hba": 2, "charge": 0}


def _compativel(pa, pd_, tol):
    return (abs(pa["mw"] - pd_["mw"]) <= tol["mw"]
            and abs(pa["logp"] - pd_["logp"]) <= tol["logp"]
            and abs(pa["hbd"] - pd_["hbd"]) <= tol["hbd"]
            and abs(pa["hba"] - pd_["hba"]) <= tol["hba"]
            and abs(pa["charge"] - pd_["charge"]) <= tol["charge"])


def parear(smiles_ativos, smiles_biblioteca, n_por_ativo=50,
           max_tanimoto=0.35, tolerancia=None, seed=0, limite_busca=4000):
    """
    Devolve (lista_decoys, relatorio).

    max_tanimoto garante que o decoy nao seja um analogo do ativo, que seria
    provavelmente ativo tambem e entraria como falso negativo.
    """
    tol = dict(TOLERANCIA)
    if tolerancia:
        tol.update(tolerancia)
    rng = random.Random(seed)

    props_ativos, fps_ativos, ativos_ok = [], [], []
    for s in smiles_ativos:
        p, f = properties(s), fingerprint(s)
        if p and f is not None:
            props_ativos.append(p)
            fps_ativos.append(f)
            ativos_ok.append(s)

    bib = list(smiles_biblioteca)
    rng.shuffle(bib)
    cache = {}

    escolhidos, usados = [], set()
    sem_decoy = 0
    for s_at, p_at, f_at in zip(ativos_ok, props_ativos, fps_ativos):
        achados, olhados = [], 0
        for s in bib:
            if olhados >= limite_busca or len(achados) >= n_por_ativo:
                break
            olhados += 1
            if s in usados:
                continue
            if s not in cache:
                cache[s] = (properties(s), fingerprint(s))
            p_d, f_d = cache[s]
            if p_d is None or f_d is None:
                continue
            if not _compativel(p_at, p_d, tol):
                continue
            if max(bulk_similarity(f_d, fps_ativos)) > max_tanimoto:
                continue
            achados.append(s)
            usados.add(s)
        if not achados:
            sem_decoy += 1
        escolhidos.extend(achados)

    rel = {
        "ativos": len(ativos_ok),
        "decoys": len(escolhidos),
        "por_ativo_pedido": n_por_ativo,
        "por_ativo_obtido": round(len(escolhidos) / max(1, len(ativos_ok)), 2),
        "ativos_sem_decoy": sem_decoy,
        "max_tanimoto": max_tanimoto,
    }
    return escolhidos, rel


def montar_conjunto(smiles_ativos, smiles_decoys):
    """Devolve (smiles, rotulos) embaralhados, prontos para ranquear."""
    pares = [(s, 1) for s in smiles_ativos] + [(s, 0) for s in smiles_decoys]
    random.Random(1).shuffle(pares)
    return [s for s, _ in pares], [r for _, r in pares]
