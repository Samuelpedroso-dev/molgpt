"""
Dois dos tres modelos de triagem.

  sintetizabilidade  regressor destilado do sascorer do RDKit, treinado nas
                     mesmas moleculas em que o gerador foi treinado
  alerta toxicologico  classificador de 12 ensaios do Tox21, split por
                       esqueleto de Murcko, que e o split honesto

Os dois comem a MESMA impressao digital que o navegador calcula.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/claude/quim")
torch.set_num_threads(2)
F = "/home/claude/exp/feat"
SAIDA = "/home/claude/exp/saida"


class Rede(nn.Module):
    def __init__(self, nin, nout, h1=128, h2=64):
        super().__init__()
        self.r = nn.Sequential(nn.Linear(nin, h1), nn.GELU(), nn.Linear(h1, h2), nn.GELU(),
                               nn.Linear(h2, nout))

    def forward(self, x):
        return self.r(x)


def confere_coluna_sas():
    """A coluna SAS do arquivo e mesmo o sascorer do RDKit? Confere em 400."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import RDConfig
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    sm = open(f"{F}/zinc_smiles.txt").read().split("\n")
    y = np.load(f"{F}/zinc_y.npy")
    rng = np.random.RandomState(0); idx = rng.choice(len(sm), 400, replace=False)
    a, b = [], []
    for i in idx:
        m = Chem.MolFromSmiles(sm[i])
        if m is None:
            continue
        a.append(sascorer.calculateScore(m)); b.append(float(y[i, 0]))
    d = np.abs(np.array(a) - np.array(b))
    print(f"coluna SAS contra sascorer do RDKit, erro medio {d.mean():.4f}, "
          f"maximo {d.max():.4f}, correlacao {np.corrcoef(a, b)[0,1]:.5f}")
    return float(d.mean())


def treina_sas():
    X = np.load(f"{F}/zinc_X.npy"); Y = np.load(f"{F}/zinc_y.npy")
    y = Y[:, 0]
    rng = np.random.RandomState(0); p = rng.permutation(len(X))
    corte = int(len(X) * .88)
    tr, te = p[:corte], p[corte:]
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
    Xe = torch.tensor(X[te]); ye = y[te]
    m = Rede(X.shape[1], 1)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    ep = 14
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    t0 = time.time()
    for e in range(ep):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            b = perm[i:i+1024]
            opt.zero_grad()
            perda = nn.functional.mse_loss(m(Xt[b]).squeeze(-1), yt[b])
            perda.backward(); opt.step()
        sch.step()
        if e % 4 == 3 or e == ep - 1:
            m.eval()
            with torch.no_grad():
                pr = m(Xe).squeeze(-1).numpy()
            print(f"  epoca {e+1:2d}  MAE {np.abs(pr-ye).mean():.3f}  "
                  f"Spearman {spearmanr(pr, ye).statistic:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    m.eval()
    with torch.no_grad():
        pr = m(Xe).squeeze(-1).numpy()
    res = dict(n_treino=len(tr), n_teste=len(te), mae=float(np.abs(pr-ye).mean()),
               spearman=float(spearmanr(pr, ye).statistic),
               dentro_de_05=float(np.mean(np.abs(pr-ye) < .5)))
    print(f"sintetizabilidade, teste {len(te)} moleculas, MAE {res['mae']:.3f}, "
          f"Spearman {res['spearman']:.4f}, dentro de 0,5 ponto {100*res['dentro_de_05']:.1f}%")
    torch.save(m.state_dict(), f"{SAIDA}/sas.pt")
    return res


def esqueleto(s):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return s
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return s


def treina_tox():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    X = np.load(f"{F}/tox_X.npy"); Y = np.load(f"{F}/tox_Y.npy")
    sm = open(f"{F}/tox_smiles.txt").read().split("\n")
    ensaios = open(f"{F}/tox_ensaios.txt").read().split("\n")
    grupos = {}
    for i, s in enumerate(sm):
        grupos.setdefault(esqueleto(s), []).append(i)
    # esqueletos maiores primeiro no treino, o teste fica com os raros, que e o caso dificil
    ordem = sorted(grupos.values(), key=len, reverse=True)
    tr, te = [], []
    for g in ordem:
        (tr if len(tr) < len(sm) * .8 else te).extend(g)
    print(f"tox21, {len(grupos)} esqueletos, treino {len(tr)}, teste {len(te)}, "
          f"sem esqueleto em comum")
    Xt = torch.tensor(X[tr]); Yt = torch.tensor(Y[tr])
    Mt = ~torch.isnan(Yt)
    Yt = torch.nan_to_num(Yt)
    Xe = torch.tensor(X[te]); Ye = Y[te]
    pos = np.nansum(Y[tr], 0); neg = np.sum(~np.isnan(Y[tr]), 0) - pos
    peso = torch.tensor((neg / np.maximum(pos, 1)).astype(np.float32))
    print("  positivos por ensaio", {e: int(p) for e, p in zip(ensaios, pos)})
    m = Rede(X.shape[1], len(ensaios))
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-4)
    ep = 60
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    for e in range(ep):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i+256]
            opt.zero_grad()
            lg = m(Xt[b])
            pe = nn.functional.binary_cross_entropy_with_logits(
                lg, Yt[b], weight=Mt[b].float(), pos_weight=peso, reduction="sum")
            (pe / Mt[b].sum().clamp(min=1)).backward()
            opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        lg = m(Xe).numpy()
    aucs = {}
    for j, e in enumerate(ensaios):
        ok = ~np.isnan(Ye[:, j])
        if ok.sum() > 20 and 0 < Ye[ok, j].sum() < ok.sum():
            aucs[e] = round(float(roc_auc_score(Ye[ok, j], lg[ok, j])), 4)
    print("  ROC AUC por ensaio", json.dumps(aucs))
    print(f"  media {np.mean(list(aucs.values())):.4f}, pior {min(aucs.values()):.4f}, "
          f"melhor {max(aucs.values()):.4f}")
    torch.save(m.state_dict(), f"{SAIDA}/tox.pt")
    return dict(auc=aucs, media=float(np.mean(list(aucs.values()))),
                n_treino=len(tr), n_teste=len(te), ensaios=ensaios)


if __name__ == "__main__":
    r = {"sas_coluna_erro": confere_coluna_sas()}
    print("\n=== sintetizabilidade ===")
    r["sas"] = treina_sas()
    print("\n=== toxicologia ===")
    r["tox"] = treina_tox()
    json.dump(r, open(f"{SAIDA}/triagem_metricas.json", "w"), indent=1)
    print("\ngravado")
