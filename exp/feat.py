"""Calcula a impressao digital propria para os tres conjuntos e guarda em disco."""
import csv, os, sys, time
import numpy as np
sys.path.insert(0, "/home/claude/quim")
from quimica import digital, parse_smiles, valida  # noqa: E402

SAIDA = "/home/claude/exp/feat"
os.makedirs(SAIDA, exist_ok=True)


def fp_lote(smiles, nbits=1024):
    X = np.zeros((len(smiles), nbits), dtype=np.float32)
    bons = []
    for i, s in enumerate(smiles):
        a, b = parse_smiles(s)
        if len(a) < 2:
            continue
        X[i] = digital(s, nbits, atoms=a, bonds=b)
        bons.append(i)
    return X, np.array(bons)


def zinc():
    t0 = time.time()
    sm, sas, logp, qed = [], [], [], []
    for l in csv.DictReader(open("/home/claude/treino/zinc250k.csv")):
        sm.append(l["smiles"].strip())
        sas.append(float(l["SAS"])); logp.append(float(l["logP"])); qed.append(float(l["qed"]))
    X, bons = fp_lote(sm)
    np.save(f"{SAIDA}/zinc_X.npy", X[bons])
    np.save(f"{SAIDA}/zinc_y.npy", np.array([[sas[i], logp[i], qed[i]] for i in bons], dtype=np.float32))
    with open(f"{SAIDA}/zinc_smiles.txt", "w") as f:
        f.write("\n".join(sm[i] for i in bons))
    print(f"zinc  {len(bons)} de {len(sm)} moleculas, {time.time()-t0:.0f}s")


def tox21():
    import urllib.request
    url = ("https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/"
           "master/latest/tox21.csv")
    dest = "/home/claude/dados_tox21/tox21.csv"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)
    linhas = list(csv.DictReader(open(dest)))
    ensaios = [c for c in linhas[0].keys() if c.startswith(("NR-", "SR-"))]
    sm = [l["smiles"].strip() for l in linhas]
    Y = np.full((len(linhas), len(ensaios)), np.nan, dtype=np.float32)
    for i, l in enumerate(linhas):
        for j, e in enumerate(ensaios):
            if l[e] not in ("", None):
                Y[i, j] = float(l[e])
    X, bons = fp_lote(sm)
    np.save(f"{SAIDA}/tox_X.npy", X[bons]); np.save(f"{SAIDA}/tox_Y.npy", Y[bons])
    with open(f"{SAIDA}/tox_smiles.txt", "w") as f:
        f.write("\n".join(sm[i] for i in bons))
    open(f"{SAIDA}/tox_ensaios.txt", "w").write("\n".join(ensaios))
    print(f"tox21 {len(bons)} de {len(sm)} moleculas, {len(ensaios)} ensaios, "
          f"rotulos presentes {100*np.mean(~np.isnan(Y[bons])):.1f}%")


if __name__ == "__main__":
    zinc(); tox21()
