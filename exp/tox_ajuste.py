"""
Ajuste do classificador do Tox21 sem olhar o teste.

Os esqueletos vao para treino, validacao e teste. Escolho a configuracao
pela validacao, depois mido o teste UMA vez. Qualquer outra ordem e enganar
a si mesmo.
"""
import json, sys, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/claude/quim")
from quimica import digital, parse_smiles  # noqa: E402
torch.set_num_threads(2)
F = "/home/claude/exp/feat"

sm = open(f"{F}/tox_smiles.txt").read().split("\n")
Y = np.load(f"{F}/tox_Y.npy")
ensaios = open(f"{F}/tox_ensaios.txt").read().split("\n")

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
grupos = {}
for i, s in enumerate(sm):
    m = Chem.MolFromSmiles(s)
    k = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else s
    grupos.setdefault(k, []).append(i)
ordem = sorted(grupos.values(), key=len, reverse=True)
tr, va, te = [], [], []
for g in ordem:
    if len(tr) < len(sm) * .7: tr.extend(g)
    elif len(va) < len(sm) * .1: va.extend(g)
    else: te.extend(g)
print(f"treino {len(tr)} validacao {len(va)} teste {len(te)}, esqueletos disjuntos")

X1024 = np.load(f"{F}/tox_X.npy")
X2048 = np.stack([digital(s, 2048) for s in sm]).astype(np.float32)

def rede(nin, nout, h1, h2, drop):
    return nn.Sequential(nn.Linear(nin, h1), nn.GELU(), nn.Dropout(drop),
                         nn.Linear(h1, h2), nn.GELU(), nn.Dropout(drop), nn.Linear(h2, nout))

def roda(X, ind_tr, ind_av, h1, h2, drop, ep, lr, semente):
    torch.manual_seed(semente)
    Xt = torch.tensor(X[ind_tr]); Yt = torch.tensor(Y[ind_tr])
    Mt = ~torch.isnan(Yt); Yt = torch.nan_to_num(Yt)
    Xa = torch.tensor(X[ind_av]); Ya = Y[ind_av]
    pos = np.nansum(Y[ind_tr], 0); neg = np.sum(~np.isnan(Y[ind_tr]), 0) - pos
    peso = torch.tensor((neg / np.maximum(pos, 1)).astype(np.float32))
    m = rede(X.shape[1], len(ensaios), h1, h2, drop)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    for e in range(ep):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i+256]
            opt.zero_grad()
            pe = nn.functional.binary_cross_entropy_with_logits(
                m(Xt[b]), Yt[b], weight=Mt[b].float(), pos_weight=peso, reduction="sum")
            (pe / Mt[b].sum().clamp(min=1)).backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        lg = m(Xa).numpy()
    aucs = {}
    for j, e in enumerate(ensaios):
        ok = ~np.isnan(Ya[:, j])
        if ok.sum() > 20 and 0 < Ya[ok, j].sum() < ok.sum():
            aucs[e] = float(roc_auc_score(Ya[ok, j], lg[ok, j]))
    return m, aucs

t0 = time.time()
cfgs = [("1024 h128 d0.0 ep60", X1024, 128, 64, 0.0, 60, 1.5e-3),
        ("1024 h256 d0.3 ep80", X1024, 256, 64, 0.3, 80, 1.5e-3),
        ("2048 h256 d0.3 ep80", X2048, 256, 64, 0.3, 80, 1.5e-3),
        ("2048 h384 d0.4 ep120", X2048, 384, 96, 0.4, 120, 1e-3)]
melhor = None
for nome, X, h1, h2, d, ep, lr in cfgs:
    _, a = roda(X, tr, va, h1, h2, d, ep, lr, 0)
    med = np.mean(list(a.values()))
    print(f"  validacao {nome:22s} AUC medio {med:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    if melhor is None or med > melhor[0]:
        melhor = (med, nome, X, h1, h2, d, ep, lr)
print(f"\nescolhida pela validacao, {melhor[1]}")
_, nome, X, h1, h2, d, ep, lr = melhor
aucs = []
for s in (0, 1, 2):
    m, a = roda(X, tr + va, te, h1, h2, d, ep, lr, s)
    aucs.append(a)
    torch.save(m.state_dict(), f"/home/claude/exp/saida/tox_s{s}.pt")
med = {e: round(float(np.mean([a[e] for a in aucs])), 4) for e in aucs[0]}
dp = {e: round(float(np.std([a[e] for a in aucs])), 4) for e in aucs[0]}
print("\nTESTE, 3 sementes")
for e in med:
    print(f"  {e:16s} {med[e]:.3f} +- {dp[e]:.3f}")
print(f"  media {np.mean(list(med.values())):.4f}")
json.dump(dict(config=nome, nbits=int(X.shape[1]), h1=h1, h2=h2, drop=d, epocas=ep, lr=lr,
               auc=med, auc_dp=dp, media=float(np.mean(list(med.values()))),
               n_treino=len(tr)+len(va), n_teste=len(te), ensaios=ensaios),
          open("/home/claude/exp/saida/tox_metricas.json", "w"), indent=1)
