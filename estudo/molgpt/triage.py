"""
Triagem de alvo. Nove criterios numericos, rodados antes de escolher.

Escolher alvo por gosto e o erro que torna todo o resto nao falseavel.
Candidato que falha em dois ou mais itens sai da lista. Se nenhum passar,
amplie a lista de candidatos, nao afrouxe os limites.
"""
from __future__ import annotations

LIMITES = {
    "n_compostos": 800,
    "n_ativos": 300,
    "faixa_log": 3.0,
    "n_arcaboucos_ativos": 20,
    "frac_tipo_dominante": 0.60,
    "n_documentos": 3,
    "frac_covalente_max": 0.30,
}

LIMIAR_ATIVO = 6.0        # pAtividade acima de 6 e melhor que 1 micromolar


def triar(linhas, tem_estrutura_pdb=None, tem_ortologo_humano=None, limites=None):
    """
    `linhas` e a saida de chem.curate. Os dois ultimos argumentos sao fatos
    que voce confere a mao no PDB e no UniProt, porque nao da para inferir
    do conjunto de bioatividade.

    Devolve (passou, relatorio). relatorio["falhas"] lista o que reprovou.
    """
    L = dict(LIMITES)
    if limites:
        L.update(limites)

    n = len(linhas)
    ativos = [l for l in linhas if l["p_activity"] >= LIMIAR_ATIVO]
    valores = [l["p_activity"] for l in linhas]
    faixa = (max(valores) - min(valores)) if valores else 0.0
    arcaboucos = {l["scaffold"] for l in ativos if l.get("scaffold")}
    docs = {d for l in linhas for d in l.get("docs", [])}
    covalentes = sum(1 for l in linhas if l.get("covalente"))
    frac_cov = covalentes / max(1, n)

    tipos = {}
    for l in linhas:
        tipos[l.get("type")] = tipos.get(l.get("type"), 0) + 1
    dominante = max(tipos.values()) / max(1, n) if tipos else 0.0

    checks = [
        ("compostos distintos", n, L["n_compostos"], n >= L["n_compostos"]),
        ("ativos acima de pAtividade 6", len(ativos), L["n_ativos"],
         len(ativos) >= L["n_ativos"]),
        ("faixa de atividade em log", round(faixa, 2), L["faixa_log"],
         faixa >= L["faixa_log"]),
        ("arcaboucos distintos entre ativos", len(arcaboucos), L["n_arcaboucos_ativos"],
         len(arcaboucos) >= L["n_arcaboucos_ativos"]),
        ("fracao do tipo dominante", round(dominante, 3), L["frac_tipo_dominante"],
         dominante >= L["frac_tipo_dominante"]),
        ("documentos de origem distintos", len(docs), L["n_documentos"],
         len(docs) >= L["n_documentos"]),
        ("fracao de covalentes", round(frac_cov, 3), L["frac_covalente_max"],
         frac_cov <= L["frac_covalente_max"]),
    ]

    if tem_estrutura_pdb is not None:
        checks.append(("estrutura no PDB com ligante nao covalente",
                       bool(tem_estrutura_pdb), True, bool(tem_estrutura_pdb)))
    if tem_ortologo_humano is not None:
        checks.append(("ortologo humano identificavel",
                       bool(tem_ortologo_humano), True, bool(tem_ortologo_humano)))

    falhas = [c[0] for c in checks if not c[3]]
    rel = {
        "criterios": [
            {"nome": nome, "valor": val, "limite": lim, "passou": ok}
            for nome, val, lim, ok in checks
        ],
        "falhas": falhas,
        "n_falhas": len(falhas),
        "veredito": "aprovado" if len(falhas) == 0 else
                    ("ressalva" if len(falhas) == 1 else "reprovado"),
    }
    return len(falhas) == 0, rel


def imprimir(rel, nome="alvo"):
    print(f"\n=== triagem de {nome} ===")
    for c in rel["criterios"]:
        marca = "ok  " if c["passou"] else "FALHA"
        print(f"  {marca} {c['nome']:<40} {c['valor']} (limite {c['limite']})")
    print(f"  veredito, {rel['veredito']}")
    if rel["falhas"]:
        print(f"  reprovou em, {', '.join(rel['falhas'])}")
