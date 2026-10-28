"""
Tabela completa de afinidade. Quatro regimes de split, cinco modelos,
tres sementes cada.

Por que quatro splits. No split aleatorio de pares, a MESMA molecula e a
MESMA proteina aparecem no treino e no teste, em outros pares. O modelo
pode acertar so por decorar a linha e a coluna. Os outros tres tiram isso.

  aleatorio       par sorteado, o facil, so serve de referencia
  molecula nova   nenhuma molecula do teste apareceu no treino
  proteina nova   nenhuma proteina do teste apareceu no treino
  os dois novos   nem a molecula nem a proteina apareceram

Por que cinco modelos. Um numero sozinho nao diz nada. A media do treino e o
piso. O modelo so de ligante mostra quanto da pontuacao sai sem NENHUMA
informacao de proteina. O vizinho mais proximo mostra quanto sai de simples
semelhanca quimica. O aditivo e o teto do que concatenar alcanca. O
interativo e a hipotese.

Tudo e medido duas vezes, com e sem os pares que estao no piso de censura
de 5,0, que sao 69,6 por cento do Davis e nao sao medidas de verdade.
"""
import json, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/claude/pkg")
sys.path.insert(0, "/home/claude/exp")
sys.path.insert(0, "/home/claude/quim")
from molgpt import metrics  # noqa: E402
from teto_proteico import carregar, desc400, ci_am  # noqa: E402
from quimica import digital  # noqa: E402

torch.set_num_threads(2)
D, H1 = 64, 128
SEMENTES = (0, 1, 2)


class SoLigante(nn.Module):
    """Nenhuma informacao de proteina entra. Mede o quanto e so a molecula."""
    def __init__(self, nd, npr):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nd, H1), nn.GELU(), nn.Linear(H1, D), nn.GELU(),
                               nn.Linear(D, 1))

    def forward(self, xd, xp):
        return self.f(xd).squeeze(-1)


class Aditivo(nn.Module):
    """y = f(molecula) + g(proteina). Nao consegue representar o par."""
    def __init__(self, nd, npr):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nd, H1), nn.GELU(), nn.Linear(H1, D), nn.GELU(),
                               nn.Linear(D, 1))
        self.g = nn.Sequential(nn.Linear(npr, H1), nn.GELU(), nn.Linear(H1, D), nn.GELU(),
                               nn.Linear(D, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, xd, xp):
        return (self.f(xd) + self.g(xp) + self.b).squeeze(-1)


class Interativo(nn.Module):
    """A cabeca ve o produto das duas codificacoes, entao representa o par."""
    def __init__(self, nd, npr):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nd, H1), nn.GELU(), nn.Linear(H1, D))
        self.g = nn.Sequential(nn.Linear(npr, H1), nn.GELU(), nn.Linear(H1, D))
        self.h = nn.Sequential(nn.Linear(D * 3, H1), nn.GELU(), nn.Dropout(0.1),
                               nn.Linear(H1, D), nn.GELU(), nn.Linear(D, 1))

    def forward(self, xd, xp):
        u, v = self.f(xd), self.g(xp)
        return self.h(torch.cat([u, v, u * v], -1)).squeeze(-1)


def treina_rede(Modelo, Xd, Xp, y, tr, te, pares, semente, epocas=40, lote=512):
    torch.manual_seed(semente)
    m = Modelo(Xd.shape[1], Xp.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    di = torch.tensor([pares[k][0] for k in tr]); pi = torch.tensor([pares[k][1] for k in tr])
    yt = torch.tensor(y[tr])
    for _ in range(epocas):
        m.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), lote):
            b = perm[i:i + lote]
            opt.zero_grad()
            nn.functional.mse_loss(m(Xd[di[b]], Xp[pi[b]]), yt[b]).backward()
            opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xd[torch.tensor([pares[k][0] for k in te])],
                 Xp[torch.tensor([pares[k][1] for k in te])]).numpy()
    return m, pred


def prediz_media(y, tr, te, pares, Td, _s):
    return np.full(len(te), float(np.mean(y[tr])), dtype=np.float32)


def prediz_vizinho(y, tr, te, pares, Td, _s, k=5):
    """
    Vizinho mais proximo por Tanimoto. Para cada par de teste olha os pares
    de treino COM A MESMA PROTEINA e faz media ponderada pelos k mais
    parecidos. Se a proteina nao apareceu no treino, cai na media do treino,
    que e exatamente a limitacao que esse baseline tem.
    """
    por_prot = {}
    for k_ in tr:
        i, j = pares[k_]
        por_prot.setdefault(j, []).append((i, y[k_]))
    base = float(np.mean(y[tr]))
    saida = np.empty(len(te), dtype=np.float32)
    for n, k_ in enumerate(te):
        i, j = pares[k_]
        viz = por_prot.get(j)
        if not viz:
            saida[n] = base; continue
        idx = np.array([v[0] for v in viz])
        sim = Td[i, idx]
        ordem = np.argsort(-sim)[:k]
        s = sim[ordem]
        if s.sum() <= 1e-9:
            saida[n] = base
        else:
            saida[n] = float(np.average([viz[o][1] for o in ordem], weights=s))
    return saida


def main():
    t0 = time.time()
    smiles, seqs, pKd = carregar()
    nd, npr = len(smiles), len(seqs)
    FP = np.stack([np.array(digital(s, 1024), dtype=np.float32) for s in smiles])
    Xd = torch.tensor(FP)
    Xp = torch.tensor(np.stack([desc400(s) for s in seqs]))
    inter = FP @ FP.T
    tam = FP.sum(1)
    Td = inter / (tam[:, None] + tam[None, :] - inter + 1e-9)   # Tanimoto entre as 68
    pares = [(i, j) for i in range(nd) for j in range(npr)]
    y = np.array([pKd[i, j] for i, j in pares], dtype=np.float32)
    piso = np.isclose(y, 5.0)
    print(f"Davis {nd} x {npr} = {len(pares)} pares, piso de censura {100*piso.mean():.1f}%")
    print(f"Tanimoto media entre as {nd} moleculas do Davis "
          f"{Td[np.triu_indices(nd,1)].mean():.3f}\n")

    def splits(semente):
        rng = np.random.RandomState(semente)
        idx = rng.permutation(len(pares)); c = int(len(idx) * .8)
        dfora = set(rng.permutation(nd)[:int(nd * .2)].tolist())
        pfora = set(rng.permutation(npr)[:int(npr * .2)].tolist())
        def sep(cond):
            return ([k for k, p in enumerate(pares) if not cond(p)],
                    [k for k, p in enumerate(pares) if cond(p)])
        return {
            "aleatorio": (idx[:c].tolist(), idx[c:].tolist()),
            "molecula nova": sep(lambda p: p[0] in dfora),
            "proteina nova": sep(lambda p: p[1] in pfora),
            "os dois novos": ([k for k, p in enumerate(pares)
                               if p[0] not in dfora and p[1] not in pfora],
                              [k for k, p in enumerate(pares)
                               if p[0] in dfora and p[1] in pfora]),
        }

    modelos = [("media do treino", prediz_media), ("vizinho Tanimoto", prediz_vizinho),
               ("so ligante", SoLigante), ("aditivo", Aditivo), ("interativo", Interativo)]
    tabela = []
    for modo in ("aleatorio", "molecula nova", "proteina nova", "os dois novos"):
        for nome, Mod in modelos:
            cis, cis_sp, sps, mses = [], [], [], []
            for s in SEMENTES:
                tr, te = splits(s)[modo]
                if isinstance(Mod, type) and issubclass(Mod, nn.Module):
                    _, pred = treina_rede(Mod, Xd, Xp, y, tr, te, pares, s)
                else:
                    pred = Mod(y, tr, te, pares, Td, s)
                r = metrics.regressao(list(y[te]), list(pred))
                mses.append(r["mse"]); sps.append(r["spearman"])
                cis.append(ci_am(y[te], pred))
                vivo = ~piso[te]
                cis_sp.append(ci_am(y[te][vivo], pred[vivo]) if vivo.sum() > 50 else float("nan"))
            l = dict(split=modo, modelo=nome, n_teste=len(te),
                     mse=round(float(np.mean(mses)), 4), spearman=round(float(np.mean(sps)), 4),
                     ci=round(float(np.mean(cis)), 4), ci_dp=round(float(np.std(cis)), 4),
                     ci_sem_piso=round(float(np.nanmean(cis_sp)), 4))
            tabela.append(l)
            print(f"  [{time.time()-t0:4.0f}s] {modo:14s} {nome:17s} n {len(te):6d}  "
                  f"MSE {l['mse']:.3f}  CI {l['ci']:.3f} +- {l['ci_dp']:.3f}  "
                  f"CI fora do piso {l['ci_sem_piso']:.3f}", flush=True)
    json.dump(tabela, open("/home/claude/exp/saida/tabela_afinidade.json", "w"), indent=1)
    print("\ngravado em saida/tabela_afinidade.json")


if __name__ == "__main__":
    main()
