"""
O ESM2 vale a pena? Teste decisivo, sem baixar nada.

A pergunta nao e qual representacao de proteina usar, e se existe sinal de
proteina que um modelo aditivo consiga usar. Duas medidas respondem isso.

1. Decomposicao de variancia. A matriz do Davis e 68 moleculas por 442
   proteinas. Da para separar quanto da variacao de pKd vem do efeito da
   molecula, quanto do efeito da proteina, e quanto e especifico do PAR.
   Se o termo de par domina, nenhum modelo que concatena features vai
   funcionar, porque concatenacao so representa efeito aditivo.

2. Teto do identificador. Trocar o descritor de proteina por um vetor
   indicador da identidade da proteina e a MELHOR representacao possivel
   para proteina vista no treino, e um ganho de informacao infinito sobre
   qualquer embedding. Se nem isso ajuda, o ESM2 nao vai ajudar.
"""
import json, math, os, sys, pickle
import numpy as np

sys.path.insert(0, "/home/claude/pkg")
from molgpt import metrics  # noqa: E402

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

DADOS = "/home/claude/dados_davis"
AA = "ACDEFGHIKLMNPQRSTVWY"


class SafeUnpickler(pickle.Unpickler):
    PERM = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"), ("numpy", "float64"), ("numpy", "float32"),
            ("builtins", "list"), ("builtins", "dict")}

    def find_class(self, mod, nome):
        if (mod, nome) in self.PERM:
            return super().find_class(mod, nome)
        raise pickle.UnpicklingError(f"bloqueado {mod}.{nome}")


def carregar():
    lig = json.load(open(f"{DADOS}/ligands_can.txt"))
    pro = json.load(open(f"{DADOS}/proteins.txt"))
    with open(f"{DADOS}/Y", "rb") as f:
        Y = np.asarray(SafeUnpickler(f, encoding="latin1").load(), dtype=float)
    return list(lig.values()), list(pro.values()), -np.log10(Y / 1e9)


def decompor(P, rotulo):
    """
    Decomposicao de dois fatores. P e a matriz molecula por proteina.
    Devolve a fracao da variancia que cabe a cada termo.
    """
    ok = ~np.isnan(P)
    g = np.nanmean(P)
    lin = np.nanmean(P, axis=1, keepdims=True) - g      # efeito da molecula
    col = np.nanmean(P, axis=0, keepdims=True) - g      # efeito da proteina
    pred = g + lin + col
    res = P - pred
    tot = np.nansum((P - g) ** 2)
    v_lin = np.nansum((np.broadcast_to(lin, P.shape)[ok]) ** 2)
    v_col = np.nansum((np.broadcast_to(col, P.shape)[ok]) ** 2)
    v_res = np.nansum(res[ok] ** 2)
    print(f"\n  {rotulo}  ({int(ok.sum())} pares)")
    print(f"    efeito da molecula      {100*v_lin/tot:5.1f}%")
    print(f"    efeito da proteina      {100*v_col/tot:5.1f}%")
    print(f"    especifico do par       {100*v_res/tot:5.1f}%")
    return dict(rotulo=rotulo, n=int(ok.sum()), molecula=round(100*v_lin/tot, 1),
                proteina=round(100*v_col/tot, 1), par=round(100*v_res/tot, 1))


def desc24(seq):
    s = seq.upper(); n = max(len(s), 1)
    kd = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,
                       1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
    carga = {"D":-1,"E":-1,"K":1,"R":1,"H":0.1}
    return np.array([s.count(a)/n for a in AA] +
                    [math.log(n), sum(kd.get(c,0) for c in s)/n,
                     sum(carga.get(c,0) for c in s)/n, sum(s.count(a) for a in "FWY")/n],
                    dtype=np.float32)


def desc400(seq):
    """Composicao de dipeptideos, 400 features. Captura vizinhanca local."""
    s = "".join(c for c in seq.upper() if c in AA)
    idx = {a: i for i, a in enumerate(AA)}
    v = np.zeros(400, dtype=np.float32)
    for i in range(len(s) - 1):
        v[idx[s[i]] * 20 + idx[s[i+1]]] += 1
    return v / max(1, len(s) - 1)


def ci_am(y, p, n=3500):
    import random
    if len(y) > n:
        r = random.Random(0); k = r.sample(range(len(y)), n)
        y = [y[i] for i in k]; p = [p[i] for i in k]
    return metrics.concordance_index(list(y), list(p))


def main():
    smiles, seqs, pKd = carregar()
    nd, npr = len(smiles), len(seqs)
    piso = np.isclose(pKd, 5.0)
    print(f"Davis, {nd} moleculas x {npr} proteinas, {pKd.size} pares")
    print(f"no piso de censura 5.0, {int(piso.sum())} ({100*piso.mean():.1f}%)")

    print("\n=== 1. DECOMPOSICAO DE VARIANCIA ===")
    dec = [decompor(pKd, "todos os pares")]
    Pm = pKd.copy(); Pm[piso] = np.nan
    dec.append(decompor(Pm, "so os medidos, sem o piso"))

    print("\n=== 2. TETO DA REPRESENTACAO DE PROTEINA ===")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    FD = np.stack([np.array(gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)), dtype=np.float32)
                   for s in smiles])
    F24 = np.stack([desc24(s) for s in seqs])
    F400 = np.stack([desc400(s) for s in seqs])
    FID = np.eye(npr, dtype=np.float32)          # identidade, o teto absoluto

    pares = [(i, j) for i in range(nd) for j in range(npr)]
    y = np.array([pKd[i, j] for i, j in pares], dtype=np.float32)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(pares))
    corte = int(len(idx) * .8)
    tr, te = idx[:corte], idx[corte:]

    def X(ind, FP):
        bl = [FD[[pares[k][0] for k in ind]]]
        if FP is not None:
            bl.append(FP[[pares[k][1] for k in ind]])
        return np.hstack(bl)

    linhas = []
    for nome, FP in [("so a molecula", None),
                     ("mais 24 descritores", F24),
                     ("mais 400 dipeptideos", F400),
                     ("mais identidade da proteina", FID)]:
        for mod, nm in [(Ridge(alpha=1.0), "ridge"),
                        (HistGradientBoostingRegressor(max_iter=160, random_state=0), "boosting")]:
            Xtr, Xte = X(tr, FP), X(te, FP)
            mod.fit(Xtr, y[tr])
            p = mod.predict(Xte)
            r = metrics.regressao(list(y[te]), list(p))
            ci = round(ci_am(y[te], p), 4)
            linhas.append(dict(repr=nome, modelo=nm, mse=r["mse"], spearman=r["spearman"], ci=ci,
                               n_features=Xtr.shape[1]))
            print(f"  {nome:30s} {nm:9s} feat {Xtr.shape[1]:5d}  "
                  f"MSE {r['mse']:.3f}  Spearman {r['spearman']:.3f}  CI {ci:.3f}", flush=True)

    os.makedirs("/home/claude/exp/saida", exist_ok=True)
    with open("/home/claude/exp/saida/teto_proteico.json", "w") as f:
        json.dump({"decomposicao": dec, "teto": linhas}, f, indent=1, default=float)
    print("\ngravado em /home/claude/exp/saida/teto_proteico.json")


if __name__ == "__main__":
    main()
