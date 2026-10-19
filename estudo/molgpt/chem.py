"""
Padronizacao de estrutura e de atividade.

Tudo que entra no projeto passa por aqui antes de virar dado de treino.
Modulo puro RDKit, sem torch, testavel sozinho.
"""
from __future__ import annotations

import math
import re

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_UNCHARGER = rdMolStandardize.Uncharger()
_LARGEST = rdMolStandardize.LargestFragmentChooser()

# Tokenizacao de SMILES, regex padrao do Molecular Transformer
SMILES_RE = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)

# Subestruturas que marcam inibidor covalente. Lista curta e deliberadamente
# conservadora, serve para separar, nao para julgar.
COVALENTES = {
    "vinil sulfona": "C=CS(=O)(=O)",
    "acrilamida": "C=CC(=O)N",
    "cloroacetamida": "ClCC(=O)N",
    "aldeido": "[CX3H1](=O)[#6]",
    "nitrila ativada": "[#6]C#N",
    "epoxido": "C1OC1",
    "boronico": "B(O)O",
}
_COV_PAT = {k: Chem.MolFromSmarts(v) for k, v in COVALENTES.items()}


def tokenize(smiles: str):
    """Tokens de um SMILES, ou None se a reconstrucao nao bater."""
    toks = SMILES_RE.findall(smiles)
    return toks if "".join(toks) == smiles else None


def standardize(smiles: str, neutralize: bool = True, keep_stereo: bool = True):
    """
    Canonicaliza, mantem o maior fragmento organico, opcionalmente neutraliza.
    Devolve None para entrada invalida ou vazia.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    # Atencao, MolFromSmiles("") devolve molecula de zero atomos, nao None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        mol = _LARGEST.choose(mol)
        if neutralize:
            mol = _UNCHARGER.uncharge(mol)
        if not keep_stereo:
            Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def inchikey(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def scaffold(smiles: str, generic: bool = False):
    """Arcabouco de Bemis Murcko. generic apaga tipo de atomo e ordem de ligacao."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            core = MurckoScaffold.MakeScaffoldGeneric(core)
        s = Chem.MolToSmiles(core)
        return s if s else None
    except Exception:
        return None


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return None if mol is None else _MORGAN.GetFingerprint(mol)


def bulk_similarity(fp, fps):
    return DataStructs.BulkTanimotoSimilarity(fp, list(fps))


def is_covalent(smiles: str):
    """Devolve a lista de alertas covalentes encontrados. Vazia significa limpo."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return []
    return [k for k, pat in _COV_PAT.items() if pat is not None and mol.HasSubstructMatch(pat)]


def properties(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    try:
        qed = QED.qed(mol)
    except Exception:
        qed = None
    return {
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "hbd": hbd,
        "hba": hba,
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "heavy": mol.GetNumHeavyAtoms(),
        "charge": Chem.GetFormalCharge(mol),
        "qed": None if qed is None else round(qed, 3),
        "lipinski": int(mw <= 500) + int(logp <= 5) + int(hbd <= 5) + int(hba <= 10),
    }


def p_activity(value, unit="nM", relation="="):
    """
    Converte atividade para pAtividade em molar.

    Devolve (valor, censurado). censurado e True quando a medida original era
    do tipo maior que ou menor que, caso em que o numero NAO deve entrar numa
    regressao comum, e sim ser tratado como censura.
    """
    if value is None:
        return None, False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, False
    if v <= 0:
        return None, False
    fator = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}.get(unit)
    if fator is None:
        return None, False
    p = -math.log10(v * fator)
    censurado = relation not in ("=", "==", None, "")
    return round(p, 4), censurado


ORDEM_PREFERIDA = ("Ki", "Kd", "IC50", "EC50")


def curate(registros, tipo_alvo=None, max_desvio_log=1.0, min_tokens=6, max_tokens=120):
    """
    Curadoria completa de um conjunto de bioatividade.

    `registros` e uma lista de dicionarios com, no minimo, as chaves
    smiles, value, unit, relation, type, year e doc.

    Devolve (linhas, relatorio). O relatorio diz quantos sairam por cada regra,
    e e o que vai para o README.
    """
    rel = {
        "entrada": len(registros),
        "smiles_invalido": 0,
        "tokenizacao": 0,
        "tamanho": 0,
        "tipo_descartado": 0,
        "atividade_invalida": 0,
        "censurado": 0,
        "duplicata_divergente": 0,
        "saida": 0,
    }

    tipos = {}
    for r in registros:
        tipos[r.get("type")] = tipos.get(r.get("type"), 0) + 1
    if tipo_alvo is None:
        for t in ORDEM_PREFERIDA:
            if tipos.get(t, 0) > 0:
                tipo_alvo = t
                break
    rel["tipos_vistos"] = tipos
    rel["tipo_escolhido"] = tipo_alvo

    por_chave = {}
    for r in registros:
        if r.get("type") != tipo_alvo:
            rel["tipo_descartado"] += 1
            continue
        smi = standardize(r.get("smiles"))
        if smi is None:
            rel["smiles_invalido"] += 1
            continue
        toks = tokenize(smi)
        if toks is None:
            rel["tokenizacao"] += 1
            continue
        if not (min_tokens <= len(toks) <= max_tokens):
            rel["tamanho"] += 1
            continue
        p, cens = p_activity(r.get("value"), r.get("unit", "nM"), r.get("relation", "="))
        if p is None:
            rel["atividade_invalida"] += 1
            continue
        if cens:
            rel["censurado"] += 1
            continue
        k = inchikey(smi) or smi
        por_chave.setdefault(k, []).append((smi, p, r.get("year"), r.get("doc")))

    linhas = []
    for k, grupo in por_chave.items():
        valores = sorted(p for _, p, _, _ in grupo)
        if len(valores) > 1 and (valores[-1] - valores[0]) > max_desvio_log:
            rel["duplicata_divergente"] += 1
            continue
        mediana = valores[len(valores) // 2]
        smi = grupo[0][0]
        anos = [a for _, _, a, _ in grupo if a]
        linhas.append({
            "inchikey": k,
            "smiles": smi,
            "p_activity": round(mediana, 4),
            "n_medidas": len(grupo),
            "desvio_log": round(valores[-1] - valores[0], 3) if len(valores) > 1 else 0.0,
            "year": min(anos) if anos else None,
            "docs": sorted({d for _, _, _, d in grupo if d}),
            "type": tipo_alvo,
            "covalente": bool(is_covalent(smi)),
            "scaffold": scaffold(smi),
        })

    rel["saida"] = len(linhas)
    rel["covalentes"] = sum(1 for l in linhas if l["covalente"])
    rel["frac_covalente"] = round(rel["covalentes"] / max(1, len(linhas)), 3)
    return linhas, rel
