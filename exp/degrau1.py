"""
Degrau 1. A interacao explicita rompe o teto aditivo de 0.790?

A medicao anterior mostrou que um modelo ADITIVO, mesmo com representacao
perfeita da proteina, chega no maximo a 0.790 de indice de concordancia,
e que 65 a 73 por cento da variancia do Davis e especifica do PAR.

Aqui os dois modelos tem a mesma capacidade e a mesma entrada. A unica
diferenca e que um soma os dois lados e o outro multiplica.

  aditivo     y = f(molecula) + g(proteina)          nao pode representar par
  interativo  y = h([u, v, u*v])                     pode

Se o interativo ganhar, a conclusao e de arquitetura, nao de representacao,
e a prioridade do projeto muda.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/claude/pkg")
sys.path.insert(0, "/home/claude/exp")
from molgpt import metrics  # noqa: E402
from teto_proteico import carregar, desc400, ci_am  # noqa: E402

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

torch.set_num_threads(2)
D = 64


class Aditivo(nn.Module):
    """y = f(molecula) + g(proteina). Aditivo por construcao."""
    def __init__(self, nd, npr):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nd, 256), nn.GELU(), nn.Linear(256, D), nn.GELU(),
                               nn.Linear(D, 1))
        self.g = nn.Sequential(nn.Linear(npr, 256), nn.GELU(), nn.Linear(256, D), nn.GELU(),
                               nn.Linear(D, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, xd, xp):
        return (self.f(xd) + self.g(xp) + self.b).squeeze(-1)


class Interativo(nn.Module):
    """Mesma capacidade nos codificadores, mas a cabeca ve o produto dos dois."""
    def __init__(self, nd, npr):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nd, 256), nn.GELU(), nn.Linear(256, D))
        self.g = nn.Sequential(nn.Linear(npr, 256), nn.GELU(), nn.Linear(256, D))
        self.h = nn.Sequential(nn.Linear(D * 3, 256), nn.GELU(), nn.Dropout(0.1),
                               nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, xd, xp):
        u, v = self.f(xd), self.g(xp)
        return self.h(torch.cat([u, v, u * v], -1)).squeeze(-1)


def treinar(Modelo, Xd, Xp, y, tr, te, pares, seed, epocas=40, lote=512):
    torch.manual_seed(seed)
    m = Modelo(Xd.shape[1], Xp.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    di = torch.tensor([pares[k][0] for k in tr]); pi = torch.tensor([pares[k][1] for k in tr])
    yt = torch.tensor(y[tr])
    dte = torch.tensor([pares[k][0] for k in te]); pte = torch.tensor([pares[k][1] for k in te])
    n = len(tr)
    for ep in range(epocas):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, lote):
            b = perm[i:i + lote]
            opt.zero_grad()
            p = m(Xd[di[b]], Xp[pi[b]])
            loss = nn.functional.mse_loss(p, yt[b])
            loss.backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xd[dte], Xp[pte]).numpy()
    r = metrics.regressao(list(y[te]), list(pred))
    return r["mse"], r["spearman"], ci_am(y[te], pred)


def main():
    t0 = time.time()
    smiles, seqs, pKd = carregar()
    nd, npr = len(smiles), len(seqs)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    Xd = torch.tensor(np.stack([np.array(gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)),
                                         dtype=np.float32) for s in smiles]))
    Xp = torch.tensor(np.stack([desc400(s) for s in seqs]))
    pares = [(i, j) for i in range(nd) for j in range(npr)]
    y = np.array([pKd[i, j] for i, j in pares], dtype=np.float32)
    print(f"Davis, {nd} x {npr}, features molecula {Xd.shape[1]}, proteina {Xp.shape[1]}")

    def splits(seed):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(pares)); c = int(len(idx) * .8)
        alea = (idx[:c].tolist(), idx[c:].tolist())
        perm = rng.permutation(npr); teste = set(perm[:int(npr * .2)].tolist())
        frio = ([k for k, (i, j) in enumerate(pares) if j not in teste],
                [k for k, (i, j) in enumerate(pares) if j in teste])
        return {"aleatorio": alea, "proteina nova": frio}

    res = []
    for nome, Mod in [("aditivo", Aditivo), ("interativo", Interativo)]:
        for modo in ("aleatorio", "proteina nova"):
            cis, mses, sps = [], [], []
            for seed in (0, 1, 2):
                tr, te = splits(seed)[modo]
                mse, sp, ci = treinar(Mod, Xd, Xp, y, tr, te, pares, seed)
                mses.append(mse); sps.append(sp); cis.append(ci)
            r = dict(modelo=nome, split=modo,
                     mse=round(float(np.mean(mses)), 4), mse_dp=round(float(np.std(mses)), 4),
                     spearman=round(float(np.mean(sps)), 4),
                     ci=round(float(np.mean(cis)), 4), ci_dp=round(float(np.std(cis)), 4))
            res.append(r)
            print(f"  [{time.time()-t0:5.0f}s] {nome:11s} {modo:15s} "
                  f"MSE {r['mse']:.3f}  Spearman {r['spearman']:.3f}  "
                  f"CI {r['ci']:.3f} +- {r['ci_dp']:.3f}", flush=True)

    os.makedirs("/home/claude/exp/saida", exist_ok=True)
    json.dump(res, open("/home/claude/exp/saida/degrau1.json", "w"), indent=1)

    print("\n=== veredito ===")
    for modo in ("aleatorio", "proteina nova"):
        a = next(r for r in res if r["modelo"] == "aditivo" and r["split"] == modo)
        i = next(r for r in res if r["modelo"] == "interativo" and r["split"] == modo)
        d = i["ci"] - a["ci"]
        ruido = max(a["ci_dp"], i["ci_dp"])
        print(f"  {modo:15s} interativo {i['ci']:.3f} contra aditivo {a['ci']:.3f}, "
              f"diferenca {d:+.3f}, ruido {ruido:.3f}  "
              f"{'ROMPE' if d > 2*ruido else 'nao rompe'}")
    print(f"  teto aditivo medido antes, 0.790")


if __name__ == "__main__":
    main()
