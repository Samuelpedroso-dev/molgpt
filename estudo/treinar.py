"""
Converge o MolGPT. Mesma arquitetura de 809.728 parametros, treinada de verdade.

A hipotese a testar e simples. A validade de 56 por cento nao vem de falta de
capacidade, vem de o modelo ter visto os dados aproximadamente uma vez. Se for
isso, treinar a MESMA arquitetura por mais passos resolve, sem aumentar nada.

Diferencas de engenharia em relacao a rodada anterior, para caber em 2 CPUs.

1. Sequencias empacotadas em vez de preenchidas com PAD. A rodada anterior
   calculava atencao sobre 80 posicoes quando a molecula mediana tem perto de
   44 tokens, ou seja, quase metade do calculo era sobre preenchimento.
2. Validade medida DURANTE o treino, que e o grafico que interessa.
3. Checkpoint e log continuos, para o resultado sobreviver a qualquer parada.
"""
import json, math, os, re, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(2)

AQUI = os.path.dirname(os.path.abspath(__file__))
KEKULE = os.environ.get("KEKULE") == "1"
SAIDA = os.path.join(AQUI, "saida_kekule" if KEKULE else "saida")
os.makedirs(SAIDA, exist_ok=True)

CFG = dict(
    n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.05,
    batch=64, lr=1.2e-3, min_lr=1.2e-4, warmup=200,
    max_iters=int(os.environ.get("ITERS", 14000)),
    aval_cada=500, n_amostras=250, n_bins=5, seed=42,
)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

TOK_RE = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- dados
def preparar():
    cache = os.path.join(SAIDA, "dados.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        log(f"cache carregado, {len(z['stream'])} tokens")
        return z["stream"], list(z["itos"]), set(z["treino_smiles"].tolist())

    import pandas as pd
    df = pd.read_csv(os.path.join(AQUI, "zinc250k.csv"))
    df["smiles"] = df["smiles"].str.strip()
    log(f"csv lido, {len(df)} linhas")

    smis, logps, qeds = [], [], []
    for s, lp, q in zip(df["smiles"], df["logP"], df["qed"]):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        if KEKULE:
            # SMILES kekulizado nao usa atomo aromatico minusculo, entao a classe
            # de falha dominante, que e kekulizacao, some por construcao.
            try:
                Chem.Kekulize(m, clearAromaticFlags=True)
                c = Chem.MolToSmiles(m, kekuleSmiles=True)
            except Exception:
                continue
        else:
            c = Chem.MolToSmiles(m)
        if not c:
            continue
        smis.append(c); logps.append(lp); qeds.append(q)
    log(f"canonicalizadas, {len(smis)}")

    NB = CFG["n_bins"]
    lp_e = np.quantile(logps, np.linspace(0, 1, NB + 1)[1:-1])
    q_e = np.quantile(qeds, np.linspace(0, 1, NB + 1)[1:-1])
    lp_b = np.digitize(logps, lp_e)
    q_b = np.digitize(qeds, q_e)

    toks_por_mol, descartadas = [], 0
    for s in smis:
        t = TOK_RE.findall(s)
        if "".join(t) != s or not (6 <= len(t) <= 110):
            descartadas += 1
            toks_por_mol.append(None)
            continue
        toks_por_mol.append(t)
    log(f"descartadas por tokenizacao ou tamanho, {descartadas}")

    cond = [f"<lp{i}>" for i in range(NB)] + [f"<q{i}>" for i in range(NB)]
    atomos = sorted({t for ts in toks_por_mol if ts for t in ts})
    itos = ["<pad>", "<bos>", "<eos>"] + cond + atomos
    stoi = {t: i for i, t in enumerate(itos)}
    BOS, EOS = 1, 2

    fluxo, guardadas = [], []
    for ts, a, b in zip(toks_por_mol, lp_b, q_b):
        if ts is None:
            continue
        fluxo.append(BOS)
        fluxo.append(stoi[f"<lp{a}>"])
        fluxo.append(stoi[f"<q{b}>"])
        fluxo.extend(stoi[t] for t in ts)
        fluxo.append(EOS)
        guardadas.append("".join(ts))

    stream = np.array(fluxo, dtype=np.uint16)
    log(f"fluxo empacotado, {len(stream)} tokens, vocabulario {len(itos)}, "
        f"{len(guardadas)} moleculas")
    np.savez_compressed(cache, stream=stream, itos=np.array(itos, dtype=object),
                        treino_smiles=np.array(guardadas, dtype=object))
    return stream, itos, set(guardadas)


# ----------------------------------------------------------------- modelo
class Bloco(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.p = h, p
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(p))
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        return x + self.mlp(self.ln2(x))


class MolGPT(nn.Module):
    def __init__(self, V, ctx, d, h, L, p):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.drop = nn.Dropout(p)
        self.blocos = nn.ModuleList([Bloco(d, h, p) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, alvos=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocos:
            x = b(x)
        logits = self.head(self.ln_f(x))
        perda = None
        if alvos is not None:
            perda = F.cross_entropy(logits.view(-1, logits.size(-1)), alvos.reshape(-1))
        return logits, perda


# -------------------------------------------------------------- amostragem
@torch.no_grad()
def amostrar(modelo, itos, stoi, n, temp=1.0, max_novos=110, cond=None):
    modelo.eval()
    BOS, EOS = 1, 2
    ini = [BOS]
    if cond is not None:
        ini.append(stoi[f"<lp{cond}>"])
    idx = torch.tensor([ini] * n, dtype=torch.long)
    pronto = torch.zeros(n, dtype=torch.bool)
    for _ in range(max_novos):
        ctx = idx[:, -CFG["block_size"]:]
        logits = modelo(ctx)[0][:, -1].float() / temp
        logits[:, 0] = -1e9
        logits[:, BOS] = -1e9
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        nxt[pronto] = 0
        idx = torch.cat([idx, nxt], 1)
        pronto |= nxt.squeeze(1) == EOS
        if pronto.all():
            break
    saida = []
    for linha in idx.tolist():
        t = []
        for x in linha[1:]:
            if x in (EOS, 0):
                break
            if itos[x].startswith("<"):
                continue
            t.append(itos[x])
        saida.append("".join(t))
    modelo.train()
    return saida


def medir(brutos, treino_set):
    validas, canon = [], []
    for s in brutos:
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumAtoms() == 0:
            continue
        c = Chem.MolToSmiles(m)
        if c:
            validas.append(s); canon.append(c)
    unicas = sorted(set(canon))
    novas = [c for c in unicas if c not in treino_set]
    n = max(1, len(brutos))
    p = len(validas) / n
    return {
        "n": len(brutos),
        "validade": round(p, 4),
        "erro_padrao": round(math.sqrt(max(p * (1 - p), 0) / n), 4),
        "unicidade": round(len(unicas) / max(1, len(validas)), 4),
        "novidade": round(len(novas) / max(1, len(unicas)), 4),
    }


# ------------------------------------------------------------------ treino
def main():
    stream, itos, treino_set = preparar()
    stoi = {t: i for i, t in enumerate(itos)}
    V, ctx = len(itos), CFG["block_size"]
    dados = torch.from_numpy(stream.astype(np.int64))
    n_val = min(200000, int(0.02 * len(dados)))
    val, tre = dados[:n_val], dados[n_val:]

    modelo = MolGPT(V, ctx, CFG["n_embd"], CFG["n_head"], CFG["n_layer"], CFG["dropout"])
    n_par = sum(p.numel() for p in modelo.parameters())
    log(f"modelo com {n_par:,} parametros, vocabulario {V}")

    opt = torch.optim.AdamW(modelo.parameters(), lr=CFG["lr"],
                            betas=(0.9, 0.95), weight_decay=0.1)

    def lote(d):
        i = torch.randint(len(d) - ctx - 1, (CFG["batch"],))
        x = torch.stack([d[j:j + ctx] for j in i])
        y = torch.stack([d[j + 1:j + 1 + ctx] for j in i])
        return x, y

    def lr_em(it):
        if it < CFG["warmup"]:
            return CFG["lr"] * (it + 1) / CFG["warmup"]
        pr = (it - CFG["warmup"]) / max(1, CFG["max_iters"] - CFG["warmup"])
        return CFG["min_lr"] + 0.5 * (CFG["lr"] - CFG["min_lr"]) * (1 + math.cos(math.pi * pr))

    hist = []
    t0 = time.time()
    modelo.train()
    for it in range(CFG["max_iters"] + 1):
        for g in opt.param_groups:
            g["lr"] = lr_em(it)

        if it % CFG["aval_cada"] == 0:
            modelo.eval()
            with torch.no_grad():
                pv = float(np.mean([modelo(*lote(val))[1].item() for _ in range(12)]))
                pt = float(np.mean([modelo(*lote(tre))[1].item() for _ in range(12)]))
            modelo.train()
            m = medir(amostrar(modelo, itos, stoi, CFG["n_amostras"]), treino_set)
            reg = dict(iter=it, perda_treino=round(pt, 4), perda_val=round(pv, 4),
                       segundos=round(time.time() - t0), **m)
            hist.append(reg)
            log(f"it {it:6d} perda {pt:.3f} val {pv:.3f} "
                f"validade {m['validade']*100:5.1f}% unic {m['unicidade']*100:5.1f}% "
                f"nov {m['novidade']*100:5.1f}%  {reg['segundos']}s")
            with open(os.path.join(SAIDA, "historico.json"), "w") as f:
                json.dump({"cfg": CFG, "n_parametros": n_par, "historico": hist}, f, indent=1)
            torch.save({"model": modelo.state_dict(), "itos": itos, "cfg": CFG, "iter": it},
                       os.path.join(SAIDA, "ckpt.pt"))

        if it == CFG["max_iters"]:
            break
        x, y = lote(tre)
        _, perda = modelo(x, y)
        opt.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()

    log("FIM")


if __name__ == "__main__":
    main()
