"""
Etapa 1 do roadmap, executada. Linhas de base obrigatorias de afinidade em Davis,
nos quatro cenarios de split.

Sem ESM2, porque o HuggingFace esta bloqueado neste ambiente. O lado da proteina
usa descritores calculados direto da sequencia, que e justamente o que a linha de
base exige. O ponto desta etapa nao e ter o melhor modelo, e ter o numero contra
o qual o modelo bom precisa competir depois.
"""
import json, math, os, pickle, random, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/pkg")
from molgpt import metrics  # noqa: E402

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

DADOS = "/home/claude/dados_davis"
FP_BITS = 1024


class SafeUnpickler(pickle.Unpickler):
    """
    O arquivo Y e um pickle baixado da internet, e desempacotar pickle de fonte
    nao confiavel executa codigo. Este desempacotador so permite o que o numpy
    precisa e levanta excecao em qualquer outra coisa.
    """
    PERMITIDO = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "float64"), ("numpy", "float32"),
        ("builtins", "list"), ("builtins", "dict"),
    }

    def find_class(self, module, name):
        if (module, name) in self.PERMITIDO:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"bloqueado, {module}.{name}")


def carregar():
    with open(f"{DADOS}/ligands_can.txt") as f:
        ligantes = json.load(f)
    with open(f"{DADOS}/proteins.txt") as f:
        proteinas = json.load(f)
    with open(f"{DADOS}/Y", "rb") as f:
        Y = SafeUnpickler(f, encoding="latin1").load()
    Y = np.asarray(Y, dtype=float)
    return list(ligantes.values()), list(proteinas.keys()), list(proteinas.values()), Y


AA = "ACDEFGHIKLMNPQRSTVWY"
# escalas simples, hidrofobicidade de Kyte Doolittle e carga
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                   1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CARGA = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}


def desc_proteina(seq):
    seq = seq.upper()
    n = max(len(seq), 1)
    comp = [seq.count(a) / n for a in AA]
    kd = sum(KD.get(c, 0.0) for c in seq) / n
    carga = sum(CARGA.get(c, 0.0) for c in seq)
    arom = sum(seq.count(a) for a in "FWY") / n
    return np.array(comp + [math.log(n), kd, carga / n, arom], dtype=np.float32)


def fp_ligante(smiles, gen):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(FP_BITS, dtype=np.float32)
    return np.array(gen.GetFingerprintAsNumPy(m), dtype=np.float32)


# ---------------------------------------------------------------- splits

def splits_quatro(n_d, n_p, seed, frac=0.2):
    """Devolve dicionario de modo para (indices de par de treino, de teste)."""
    rng = np.random.RandomState(seed)
    pares = [(i, j) for i in range(n_d) for j in range(n_p)]

    d_perm = rng.permutation(n_d)
    p_perm = rng.permutation(n_p)
    d_teste = set(d_perm[: max(1, int(n_d * frac))].tolist())
    p_teste = set(p_perm[: max(1, int(n_p * frac))].tolist())

    out = {}
    idx = np.arange(len(pares))
    rng.shuffle(idx)
    corte = int(len(idx) * (1 - frac))
    out["aleatorio"] = (idx[:corte].tolist(), idx[corte:].tolist())

    def por(cond_teste, cond_treino):
        tr = [k for k, (i, j) in enumerate(pares) if cond_treino(i, j)]
        te = [k for k, (i, j) in enumerate(pares) if cond_teste(i, j)]
        return tr, te

    out["molecula nova"] = por(lambda i, j: i in d_teste, lambda i, j: i not in d_teste)
    out["proteina nova"] = por(lambda i, j: j in p_teste, lambda i, j: j not in p_teste)
    out["ambos novos"] = por(
        lambda i, j: i in d_teste and j in p_teste,
        lambda i, j: i not in d_teste and j not in p_teste)
    return pares, out, d_teste, p_teste


def ci_amostrado(y, p, n=4000, seed=0):
    if len(y) > n:
        rng = random.Random(seed)
        idx = rng.sample(range(len(y)), n)
        y = [y[i] for i in idx]
        p = [p[i] for i in idx]
    return metrics.concordance_index(y, p)


def avaliar(y, p, seed=0):
    r = metrics.regressao(list(y), list(p))
    r["ci"] = round(ci_amostrado(list(y), list(p), seed=seed), 4)
    return r


def main():
    t0 = time.time()
    smiles, nomes_p, seqs, Y = carregar()
    n_d, n_p = len(smiles), len(seqs)
    print(f"Davis, {n_d} moleculas, {n_p} proteinas, matriz {Y.shape}")

    # Kd em nM para pKd. O piso de 10000 nM vira exatamente 5.0 e NAO e medida real.
    pKd = -np.log10(Y / 1e9)
    piso = np.isclose(pKd, 5.0)
    print(f"pares totais {pKd.size}, exatamente no piso 5.0, {int(piso.sum())} "
          f"({100*piso.mean():.1f}%)")
    print(f"pKd fora do piso, min {pKd[~piso].min():.2f}, max {pKd[~piso].max():.2f}, "
          f"mediana {np.median(pKd[~piso]):.2f}")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    FD = np.stack([fp_ligante(s, gen) for s in smiles])
    FP = np.stack([desc_proteina(s) for s in seqs])
    print(f"features, ligante {FD.shape}, proteina {FP.shape}")

    resultados = []
    for seed in (0, 1, 2):
        pares, conj, d_te, p_te = splits_quatro(n_d, n_p, seed)
        yv = np.array([pKd[i, j] for i, j in pares], dtype=np.float32)

        for modo, (tr, te) in conj.items():
            if len(te) < 50 or len(tr) < 200:
                continue
            ytr, yte = yv[tr], yv[te]

            def X(indices, usa_lig=True, usa_prot=True):
                blocos = []
                if usa_lig:
                    blocos.append(FD[[pares[k][0] for k in indices]])
                if usa_prot:
                    blocos.append(FP[[pares[k][1] for k in indices]])
                return np.hstack(blocos)

            modelos = {}
            modelos["media do treino"] = np.full(len(te), ytr.mean(), dtype=np.float32)

            for nome, (ul, up) in [("so molecula", (True, False)),
                                   ("so proteina", (False, True)),
                                   ("completo ridge", (True, True))]:
                m = Ridge(alpha=1.0)
                m.fit(X(tr, ul, up), ytr)
                modelos[nome] = m.predict(X(te, ul, up))

            g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                              max_depth=None, random_state=seed)
            g.fit(X(tr), ytr)
            modelos["completo boosting"] = g.predict(X(te))

            for nome, pred in modelos.items():
                r = avaliar(yte, pred, seed=seed)
                r.update(modelo=nome, split=modo, seed=seed,
                         n_treino=len(tr), n_teste=len(te))
                resultados.append(r)
            print(f"  [{time.time()-t0:5.0f}s] seed {seed} {modo:15s} "
                  f"treino {len(tr):6d} teste {len(te):6d} ok")

    os.makedirs("/home/claude/exp/saida", exist_ok=True)
    with open("/home/claude/exp/saida/davis_baselines.json", "w") as f:
        json.dump(resultados, f, indent=1, default=float)

    print("\n" + "=" * 86)
    print(f"{'split':16s} {'modelo':20s} {'MSE':>14s} {'Spearman':>14s} {'CI':>14s}")
    print("=" * 86)
    ordem_split = ["aleatorio", "molecula nova", "proteina nova", "ambos novos"]
    ordem_mod = ["media do treino", "so molecula", "so proteina",
                 "completo ridge", "completo boosting"]
    for sp in ordem_split:
        for mo in ordem_mod:
            sel = [r for r in resultados if r["split"] == sp and r["modelo"] == mo]
            if not sel:
                continue
            def ms(k):
                v = [r[k] for r in sel]
                return f"{np.mean(v):7.3f}+-{np.std(v):.3f}"
            print(f"{sp:16s} {mo:20s} {ms('mse'):>14s} {ms('spearman'):>14s} {ms('ci'):>14s}")
        print("-" * 86)
    print(f"tempo total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
