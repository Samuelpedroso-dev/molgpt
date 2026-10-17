# MolGPT, script version. Run python molgpt.py on a machine with a GPU, or python molgpt.py cpu for the small preset.
# Dependencies, pip install rdkit torch pandas numpy matplotlib

# ===== 02_config.py =====
import os, sys, re, json, math, time, random, urllib.request
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, Draw, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
RDLogger.DisableLog("rdApp.*")

SMOKE = os.environ.get("MOLGPT_SMOKE") == "1"   # quick CPU smoke test
device = "cuda" if torch.cuda.is_available() else "cpu"

CFG = dict(
    n_mols     = 12000 if SMOKE else 250000,
    block_size = 80,                       # the longest ZINC250k SMILES has 72 tokens plus 4 special ones
    n_layer    = 2   if SMOKE else 6,
    n_head     = 4   if SMOKE else 8,
    n_embd     = 64  if SMOKE else 256,
    dropout    = 0.1,
    batch      = 64  if SMOKE else 256,
    max_iters  = int(os.environ.get("MOLGPT_ITERS", 120)) if SMOKE else 6000,
    lr         = 1e-3 if SMOKE else 6e-4,
    warmup     = 20  if SMOKE else 200,
    n_bins     = 5,                        # logP and QED bins used as conditions
    seed       = 42,
)
# python molgpt.py cpu uses the small preset of the demo model, with 0.8 million parameters
PRESET = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else ""
if PRESET == "cpu" and not SMOKE:
    CFG.update(n_layer=4, n_head=4, n_embd=128, batch=128, max_iters=2600, lr=1e-3, warmup=100)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
print("device", device, "| smoke test mode" if SMOKE else "| full mode")

# ===== 03_dados.py =====
DATA_URL  = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/"
             "models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")
DATA_PATH = os.environ.get("MOLGPT_DATA", "zinc250k.csv")
if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

df = pd.read_csv(DATA_PATH)
df["smiles"] = df["smiles"].str.strip()
df = df.sample(n=min(CFG["n_mols"], len(df)), random_state=0).reset_index(drop=True)

def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None

t0 = time.time()
df["smiles"] = df["smiles"].map(canon)
df = df.dropna(subset=["smiles"]).drop_duplicates("smiles").reset_index(drop=True)
TRAIN_SET = set(df["smiles"])
print(f"{len(df)} canonical molecules in {time.time()-t0:.0f}s")
print(df[["logP", "qed"]].describe().loc[["mean", "std", "min", "max"]].round(2))

# ===== 04_tokens.py =====
TOK_RE = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")
def tokenize(s):
    toks = TOK_RE.findall(s)
    assert "".join(toks) == s, s
    return toks

NB = CFG["n_bins"]
cond_tokens = [f"<lp{i}>" for i in range(NB)] + [f"<q{i}>" for i in range(NB)]
atom_tokens = sorted({t for s in df["smiles"] for t in tokenize(s)})
itos = ["<pad>", "<bos>", "<eos>"] + cond_tokens + atom_tokens
stoi = {t: i for i, t in enumerate(itos)}
PAD, BOS, EOS = 0, 1, 2
V = len(itos)
L = CFG["block_size"]

# logP and QED bins by quantile, used as control tokens
lp_edges = np.quantile(df["logP"], np.linspace(0, 1, NB + 1)[1:-1])
q_edges  = np.quantile(df["qed"],  np.linspace(0, 1, NB + 1)[1:-1])
lp_bin = np.digitize(df["logP"], lp_edges)
q_bin  = np.digitize(df["qed"],  q_edges)

def encode(smi, lp, q):
    ids = [BOS, stoi[f"<lp{lp}>"], stoi[f"<q{q}>"]] + [stoi[t] for t in tokenize(smi)] + [EOS]
    return ids + [PAD] * (L - len(ids))

data = np.array([encode(s, a, b) for s, a, b in zip(df["smiles"], lp_bin, q_bin)], dtype=np.uint8)
n_val = max(500, int(0.02 * len(data)))
perm = np.random.permutation(len(data))
val_data   = torch.from_numpy(data[perm[:n_val]]).to(device)
train_data = torch.from_numpy(data[perm[n_val:]]).to(device)
print("vocabulary", V, "| train", len(train_data), "| validation", len(val_data))
print("example", " ".join(itos[i] for i in data[0] if i != PAD))

# ===== 05_modelo.py =====
class Block(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.p = h, p
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv  = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp  = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))
        self.drop = nn.Dropout(p)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        return x + self.mlp(self.ln2(x))

class MolGPT(nn.Module):
    def __init__(self, vocab, block, d, h, n_layer, p):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.drop = nn.Dropout(p)
        self.blocks = nn.ModuleList([Block(d, h, p) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight          # tied weights
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD)
        return logits, loss

model = MolGPT(V, L, CFG["n_embd"], CFG["n_head"], CFG["n_layer"], CFG["dropout"]).to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")

# ===== 06_treino.py =====
use_amp = device == "cuda"
amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], betas=(0.9, 0.95), weight_decay=0.1)

def get_batch(split, bs):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d), (bs,), device=device)
    seq = d[ix].long()
    return seq[:, :-1], seq[:, 1:]

def lr_at(it):
    if it < CFG["warmup"]:
        return CFG["lr"] * (it + 1) / CFG["warmup"]
    prog = (it - CFG["warmup"]) / max(1, CFG["max_iters"] - CFG["warmup"])
    return 0.1 * CFG["lr"] + 0.9 * CFG["lr"] * 0.5 * (1 + math.cos(math.pi * prog))

@torch.no_grad()
def eval_loss(n=20):
    model.eval(); out = {}
    for split in ("train", "val"):
        ls = []
        for _ in range(n):
            x, y = get_batch(split, CFG["batch"])
            with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                ls.append(model(x, y)[1].item())
        out[split] = float(np.mean(ls))
    model.train(); return out

hist = []
model.train(); t0 = time.time()
for it in range(CFG["max_iters"] + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    if it % (20 if SMOKE else 250) == 0:
        e = eval_loss(5 if SMOKE else 20); hist.append((it, e["train"], e["val"]))
        print(f"iter {it:5d} | train {e['train']:.3f} | validation {e['val']:.3f} | {time.time()-t0:.0f}s")
    if it == CFG["max_iters"]:
        break
    x, y = get_batch("train", CFG["batch"])
    with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()

torch.save({"model": model.state_dict(), "cfg": CFG, "itos": itos,
            "lp_edges": lp_edges, "q_edges": q_edges}, "molgpt.pt")
h = np.array(hist)
plt.figure(figsize=(6, 3.2))
plt.plot(h[:, 0], h[:, 1], label="train"); plt.plot(h[:, 0], h[:, 2], label="validation")
plt.xlabel("iteration"); plt.ylabel("loss"); plt.legend(); plt.title("Learning curve")
plt.tight_layout(); plt.show()

# ===== 07_gerar.py =====
@torch.no_grad()
def generate(n, cond=None, temperature=1.0, top_k=None, chunk=1000):
    """Generate n SMILES. cond is a tuple (logP bin, QED bin) and either item can be None."""
    model.eval(); out = []
    for i in range(0, n, chunk):
        b = min(chunk, n - i)
        idx = torch.full((b, 1), BOS, device=device)
        if cond is not None:
            if cond[0] is not None:
                idx = torch.cat([idx, torch.full((b, 1), stoi[f"<lp{cond[0]}>"], device=device)], 1)
                if cond[1] is not None:
                    idx = torch.cat([idx, torch.full((b, 1), stoi[f"<q{cond[1]}>"], device=device)], 1)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        while idx.shape[1] < L and not done.all():
            with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                logits = model(idx)[0][:, -1].float() / temperature
            logits[:, PAD] = -1e9; logits[:, BOS] = -1e9
            if top_k:
                v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = -1e9
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            nxt[done] = PAD
            idx = torch.cat([idx, nxt], 1); done |= nxt.squeeze(1) == EOS
        for row in idx.tolist():
            toks = []
            for t in row[1:]:
                if t in (EOS, PAD): break
                if itos[t].startswith("<"): continue
                toks.append(itos[t])
            out.append("".join(toks))
    return out

PAINS = FilterCatalog(FilterCatalogParams(FilterCatalogParams.FilterCatalogs.PAINS))
morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def props(m):
    return dict(qed=QED.qed(m), logp=Descriptors.MolLogP(m), mw=Descriptors.MolWt(m),
                hbd=Descriptors.NumHDonors(m), hba=Descriptors.NumHAcceptors(m),
                pains=PAINS.HasMatch(m))

def lipinski_ok(p):
    return p["mw"] <= 500 and p["logp"] <= 5 and p["hbd"] <= 5 and p["hba"] <= 10

def evaluate(smiles, name="sample"):
    mols = [(s, Chem.MolFromSmiles(s)) for s in smiles]
    valid = [(s, m) for s, m in mols if m is not None and len(s) > 0]
    can = [Chem.MolToSmiles(m) for _, m in valid]
    uniq = sorted(set(can))
    novel = [s for s in uniq if s not in TRAIN_SET]
    rng = random.Random(0)
    sub = rng.sample(uniq, min(1000, len(uniq)))
    fps = [morgan.GetFingerprint(Chem.MolFromSmiles(s)) for s in sub]
    sims = [DataStructs.BulkTanimotoSimilarity(f, fps[i+1:]) for i, f in enumerate(fps[:-1])]
    div = 1 - float(np.mean(np.concatenate(sims))) if sims else float("nan")
    P = [props(Chem.MolFromSmiles(s)) for s in sub]
    r = dict(sample=name, generated=len(smiles),
             validity=len(valid) / max(1, len(smiles)),
             uniqueness=len(uniq) / max(1, len(valid)),
             novelty=len(novel) / max(1, len(uniq)),
             diversity=div,
             mean_qed=float(np.mean([p["qed"] for p in P])),
             mean_logp=float(np.mean([p["logp"] for p in P])),
             lipinski=float(np.mean([lipinski_ok(p) for p in P])),
             pains=float(np.mean([p["pains"] for p in P])))
    return r, uniq, novel

N_GEN = 600 if SMOKE else (4000 if PRESET == "cpu" else 10000)
t0 = time.time()
gen_raw = generate(N_GEN, temperature=1.0)
res, gen_unique, gen_novel = evaluate(gen_raw, "GPT unconditioned")
ref_smiles = random.Random(1).sample(sorted(TRAIN_SET), min(3000, len(TRAIN_SET)))
ref, _, _ = evaluate(ref_smiles, "real ZINC")
print(f"generated {N_GEN} in {time.time()-t0:.0f}s")
print(pd.DataFrame([res, ref]).set_index("sample").round(3).T)

# property distributions against real ZINC
sample_props = [props(Chem.MolFromSmiles(s)) for s in gen_unique[:3000]]
ref_props    = [props(Chem.MolFromSmiles(s)) for s in ref_smiles]
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for a_, key, lab in zip(ax, ["logp", "qed", "mw"], ["logP", "QED", "molecular weight"]):
    a_.hist([p[key] for p in ref_props], bins=40, alpha=.55, density=True, label="real ZINC")
    a_.hist([p[key] for p in sample_props], bins=40, alpha=.55, density=True, label="GPT")
    a_.set_xlabel(lab)
ax[0].legend(); plt.suptitle("Did the GPT learn the chemical distribution?"); plt.tight_layout(); plt.show()

# does conditioning work? generate per logP bin and measure the real logP
rows = []
for b in range(NB):
    g = [s for s in generate(150 if SMOKE else 600, cond=(b, None)) if Chem.MolFromSmiles(s)]
    lp = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in g]
    rows.append((b, np.mean(lp) if lp else np.nan, len(g)))
print("logP bin requested, mean logP obtained, valid molecules")
for r_ in rows: print(r_[0], round(r_[1], 2), r_[2])

# ===== 08_julgar.py =====
# The GPT proposes and the judges filter. Here the judges are medicinal chemistry rules.
# When the affinity model exists it joins the judge function as one more judge.
def judge(smi):
    m = Chem.MolFromSmiles(smi)
    p = props(m)
    ok = lipinski_ok(p) and not p["pains"] and 200 <= p["mw"] <= 500
    return ok, p

pool = [s for s in gen_novel if judge(s)[0]]
fps_pool = [morgan.GetFingerprint(Chem.MolFromSmiles(s)) for s in pool]
ranked = sorted(range(len(pool)), key=lambda i: -props(Chem.MolFromSmiles(pool[i]))["qed"])
print(f"{len(gen_novel)} novel, {len(pool)} passed the filters")

# pick diverse candidates, avoiding near duplicates
N_SHOW = 40 if SMOKE else 120
chosen = []
for i in ranked:
    if all(DataStructs.TanimotoSimilarity(fps_pool[i], fps_pool[j]) < 0.75 for j in chosen):
        chosen.append(i)
    if len(chosen) >= N_SHOW:
        break
candidates = [pool[i] for i in chosen]
if len(candidates) < 10:   # model still weak, fill with real molecules just so the visual works
    print("few valid candidates, filling with real ZINC molecules")
    candidates += ref_smiles[:N_SHOW - len(candidates)]
print(f"{len(candidates)} molecules for the visual")
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in candidates[:12]], molsPerRow=4,
                     subImgSize=(260, 200),
                     legends=[f"QED {QED.qed(Chem.MolFromSmiles(s)):.2f}" for s in candidates[:12]])

# ===== 09_visual.py =====
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Cluster import Butina

VIZ_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Molecule map</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--tx:#e6edf3;--mut:#8b949e;--acc:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 18px 6px}
h1{margin:0;font-size:18px;font-weight:600}
header p{margin:2px 0 0;color:var(--mut);font-size:13px}
.wrap{display:flex;gap:14px;padding:10px 18px 18px;flex-wrap:wrap}
#stage{flex:1 1 560px;min-width:300px;height:640px;position:relative;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
canvas#g{width:100%;height:100%;display:block;cursor:grab}
#tip{position:absolute;pointer-events:none;background:#000c;border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px;display:none;max-width:260px;word-break:break-all}
#hint{position:absolute;left:12px;bottom:10px;color:var(--mut);font-size:12px}
#panel{flex:0 1 350px;min-width:300px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:10px}
#panel h2{margin:0;font-size:15px;font-weight:600}
.box{background:#0d1117;border:1px solid var(--line);border-radius:10px;overflow:hidden;position:relative}
#d2 svg{width:100%;height:auto;display:block}
#c3{width:100%;height:230px;display:block;cursor:grab}
.lab{position:absolute;left:8px;top:6px;font-size:11px;color:var(--mut)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.stats div{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:6px 8px;text-align:center}
.stats b{display:block;font-size:15px}.stats span{font-size:11px;color:var(--mut)}
#smi{font:12px ui-monospace,Menlo,Consolas,monospace;word-break:break-all;color:#a5d6ff;min-height:34px}
button{background:#21262d;color:var(--tx);border:1px solid var(--line);border-radius:8px;padding:5px 10px;cursor:pointer;font-size:12px}
button:hover{border-color:var(--acc)}
.nb{display:flex;gap:6px;flex-wrap:wrap}
.nb button b{color:var(--acc)}
</style></head><body>
<header><h1 id="ttl"></h1>
<p>Each dot is a molecule. Lines connect similar molecules (Tanimoto similarity) and colors show clusters. Click a dot to watch the structure build itself and to see the 3D model.</p></header>
<div class="wrap">
 <div id="stage"><canvas id="g"></canvas><div id="tip"></div>
  <div id="hint">drag a dot to move the network</div></div>
 <div id="panel">
  <h2 id="ph">Pick a molecule</h2>
  <div class="box" id="d2"><span class="lab">2D structure</span></div>
  <div class="stats"><div><b id="sq">-</b><span>QED</span></div><div><b id="sl">-</b><span>logP</span></div><div><b id="sm">-</b><span>mass</span></div></div>
  <div id="smi"></div>
  <div class="box"><span class="lab">3D structure (drag to rotate)</span><canvas id="c3"></canvas></div>
  <div class="nb" id="nb"></div>
  <div><button id="replay">replay animation</button></div>
 </div>
</div>
<script>
const D = __DATA__;
document.getElementById('ttl').textContent = D.title;
const M = D.mols, N = M.length;
const $ = id => document.getElementById(id);
const col = i => M[i].cl < 0 ? '#6e7681' : 'hsl(' + ((M[i].cl * 53 + 200) % 360) + ',72%,62%)';
const nbrs = M.map(() => []);
D.edges.forEach(e => { nbrs[e[0]].push([e[1], e[2]]); nbrs[e[1]].push([e[0], e[2]]); });
nbrs.forEach(a => a.sort((x, y) => y[1] - x[1]));
const edgeOrder = D.edges.map((e, k) => k).sort((a, b) => D.edges[b][2] - D.edges[a][2]);
const edgeRank = {}; edgeOrder.forEach((k, r) => edgeRank[k] = r);

/* ---------- rede ---------- */
const cv = $('g'), cx = cv.getContext('2d'); let W = 0, H = 0, dpr = 1;
function resize(){ dpr = window.devicePixelRatio || 1; const r = cv.getBoundingClientRect(); W = r.width; H = r.height;
  cv.width = W * dpr; cv.height = H * dpr; cx.setTransform(dpr, 0, 0, dpr, 0, 0); }
window.addEventListener('resize', resize); resize();
const nc = Math.max(1, Math.max(...M.map(m => m.cl)) + 1);
const P = M.map((m, i) => { const a = m.cl < 0 ? Math.random() * 6.283 : 2 * Math.PI * m.cl / nc + (Math.random() - .5) * .6, r = 160 + Math.random() * 120;
  return {x: Math.cos(a) * r, y: Math.sin(a) * r, vx: 0, vy: 0, fixed: false}; });
let alpha = 1, sel = -1, hov = -1, drag = -1, t0 = performance.now();
let view = {s: 1, ox: 0, oy: 0};
function sim(){
  const k = 80;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    let dx = P[i].x - P[j].x, dy = P[i].y - P[j].y, d2 = dx * dx + dy * dy + 0.01, d = Math.sqrt(d2);
    const f = (k * k) / d2 * alpha * 1.6; dx /= d; dy /= d;
    P[i].vx += dx * f; P[i].vy += dy * f; P[j].vx -= dx * f; P[j].vy -= dy * f; }
  D.edges.forEach(e => { const a = P[e[0]], b = P[e[1]]; let dx = b.x - a.x, dy = b.y - a.y; const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
    const rest = k * (1.5 - e[2]), f = (d - rest) * 0.06 * (0.4 + e[2]); dx /= d; dy /= d;
    a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f; });
  P.forEach(p => { p.vx -= p.x * 0.004; p.vy -= p.y * 0.004;
    if (!p.fixed) { p.x += p.vx; p.y += p.vy; } p.vx *= 0.82; p.vy *= 0.82; });
  alpha = Math.max(0.03, alpha * 0.992);
}
function fit(){
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; P.forEach(p => { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); });
  const s = Math.min((W - 70) / Math.max(40, x1 - x0), (H - 70) / Math.max(40, y1 - y0)), ox = W / 2 - (x0 + x1) / 2 * s, oy = H / 2 - (y0 + y1) / 2 * s;
  view.s += (s - view.s) * 0.08; view.ox += (ox - view.ox) * 0.08; view.oy += (oy - view.oy) * 0.08; }
const sx = p => p.x * view.s + view.ox, sy = p => p.y * view.s + view.oy;
const rad = i => (4 + 9 * Math.max(0, Math.min(1, (M[i].qed - 0.35) / 0.6)));
function draw(){
  const T = Math.min(1, (performance.now() - t0) / 4200);
  cx.clearRect(0, 0, W, H);
  const E = D.edges.length;
  D.edges.forEach((e, k) => { const delay = 0.55 * edgeRank[k] / Math.max(1, E), f = Math.max(0, Math.min(1, (T - 0.1 - delay) / 0.35)); if (f <= 0) return;
    const a = P[e[0]], b = P[e[1]], x0 = sx(a), y0 = sy(a), x1 = x0 + (sx(b) - x0) * f, y1 = y0 + (sy(b) - y0) * f;
    const on = sel >= 0 && (e[0] === sel || e[1] === sel), dim = sel >= 0 && !on;
    cx.strokeStyle = on ? 'rgba(88,166,255,.95)' : 'rgba(139,148,158,' + (dim ? 0.08 : 0.15 + 0.5 * e[2]) + ')';
    cx.lineWidth = on ? 2.2 : 0.7 + 1.6 * e[2]; cx.beginPath(); cx.moveTo(x0, y0); cx.lineTo(x1, y1); cx.stroke(); });
  for (let i = 0; i < N; i++) { const sc = Math.max(0, Math.min(1, (T * 1.25 - 0.1 - 0.15 * i / N) / 0.12)); if (sc <= 0) continue;
    const r = rad(i) * sc * (i === sel || i === hov ? 1.35 : 1), x = sx(P[i]), y = sy(P[i]);
    const isN = sel >= 0 && (i === sel || nbrs[sel].some(n => n[0] === i));
    cx.globalAlpha = sel >= 0 && !isN ? 0.35 : 1; cx.fillStyle = col(i); cx.beginPath(); cx.arc(x, y, r, 0, 7); cx.fill();
    if (i === sel) { cx.strokeStyle = '#fff'; cx.lineWidth = 2; cx.stroke(); } cx.globalAlpha = 1; }
}
function pick(mx, my){ let best = -1, bd = 1e9; for (let i = 0; i < N; i++) { const dx = sx(P[i]) - mx, dy = sy(P[i]) - my, d = Math.sqrt(dx * dx + dy * dy); if (d < rad(i) + 6 && d < bd) { bd = d; best = i; } } return best; }
function pos(ev){ const r = cv.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; }
cv.addEventListener('mousemove', ev => { const [mx, my] = pos(ev);
  if (drag >= 0) { P[drag].x = (mx - view.ox) / view.s; P[drag].y = (my - view.oy) / view.s; alpha = Math.max(alpha, 0.3); return; }
  hov = pick(mx, my); cv.style.cursor = hov >= 0 ? 'pointer' : 'grab'; const tip = $('tip');
  if (hov >= 0) { tip.style.display = 'block'; tip.style.left = Math.min(W - 270, mx + 14) + 'px'; tip.style.top = (my + 14) + 'px';
    tip.textContent = 'QED ' + M[hov].qed.toFixed(2) + ' | logP ' + M[hov].logp.toFixed(1) + ' | ' + M[hov].smi; } else tip.style.display = 'none'; });
cv.addEventListener('mousedown', ev => { const [mx, my] = pos(ev), i = pick(mx, my); if (i >= 0) { drag = i; P[i].fixed = true; cv._moved = false; } });
window.addEventListener('mouseup', () => { if (drag >= 0) { P[drag].fixed = false; drag = -1; } });
cv.addEventListener('click', ev => { const [mx, my] = pos(ev), i = pick(mx, my); if (i >= 0) select(i); });
function loop(){ sim(); fit(); draw(); requestAnimationFrame(loop); } loop();

/* ---------- painel 2D com animacao ---------- */
function anim2d(){
  const svg = $('d2').querySelector('svg'); if (!svg) return;
  const els = [...svg.querySelectorAll('[class]')].filter(e => /bond-|atom-/.test(e.getAttribute('class')));
  els.forEach((e, k) => { const isBond = /bond-/.test(e.getAttribute('class')); let len = 0;
    try { len = e.getTotalLength ? e.getTotalLength() : 0; } catch (x) { len = 0; }
    const delay = 120 + k * 55;
    if (isBond && len > 0) { e.style.strokeDasharray = len; e.animate([{strokeDashoffset: len, opacity: 0.2}, {strokeDashoffset: 0, opacity: 1}], {delay, duration: 380, fill: 'both', easing: 'ease-out'}); }
    else e.animate([{opacity: 0, transform: 'scale(.4)'}, {opacity: 1, transform: 'scale(1)'}], {delay: delay + 80, duration: 320, fill: 'both'}); });
}
let typeTimer = null;
function typeSmi(s){ clearInterval(typeTimer); const el = $('smi'); el.textContent = ''; let k = 0;
  const toks = s.match(/(\[[^\]]+\]|Br|Cl|%\d\d|.)/g) || [s];
  typeTimer = setInterval(() => { el.textContent += toks[k++] || ''; if (k >= toks.length) clearInterval(typeTimer); }, 55); }

/* ---------- 3D proprio, sem bibliotecas ---------- */
const CPK = {C: '#aab4c0', N: '#5b8def', O: '#ff5d5d', S: '#f5d547', F: '#7bd88f', Cl: '#3ddc84', Br: '#d2784c', I: '#a66bd1', P: '#ff9f43', H: '#ffffff'};
const RAD = {C: .34, N: .33, O: .31, S: .46, F: .28, Cl: .44, Br: .5, I: .56, P: .46, H: .2};
const c3 = $('c3'), x3 = c3.getContext('2d'); let w3 = 0, h3 = 0, mol3 = null, rotY = 0, rotX = -0.35, drag3 = null, auto = true;
function resize3(){ const r = c3.getBoundingClientRect(); w3 = r.width; h3 = r.height; c3.width = w3 * dpr; c3.height = h3 * dpr; x3.setTransform(dpr, 0, 0, dpr, 0, 0); }
window.addEventListener('resize', resize3); resize3();
function parseMB(mb){
  const L = mb.split('\n'), na = parseInt(L[3].substr(0, 3)), nb = parseInt(L[3].substr(3, 3)), at = [], bd = [];
  for (let i = 0; i < na; i++) { const l = L[4 + i]; at.push({x: +l.substr(0, 10), y: +l.substr(10, 10), z: +l.substr(20, 10), e: l.substr(31, 3).trim()}); }
  for (let i = 0; i < nb; i++) { const l = L[4 + na + i]; bd.push([parseInt(l.substr(0, 3)) - 1, parseInt(l.substr(3, 3)) - 1, parseInt(l.substr(6, 3))]); }
  const cxm = at.reduce((s, a) => s + a.x, 0) / na, cym = at.reduce((s, a) => s + a.y, 0) / na, czm = at.reduce((s, a) => s + a.z, 0) / na;
  at.forEach(a => { a.x -= cxm; a.y -= cym; a.z -= czm; });
  const R = Math.max(...at.map(a => Math.hypot(a.x, a.y, a.z))) + 0.8; return {at, bd, R}; }
function draw3(){
  x3.clearRect(0, 0, w3, h3);
  if (!mol3) { x3.fillStyle = '#8b949e'; x3.font = '12px system-ui'; x3.textAlign = 'center'; x3.fillText('no 3D conformer', w3 / 2, h3 / 2); return; }
  if (auto && !drag3) rotY += 0.012;
  const cy = Math.cos(rotY), sy_ = Math.sin(rotY), cxr = Math.cos(rotX), sxr = Math.sin(rotX), sc = Math.min(w3, h3) / 2 / mol3.R * 1.1, F = 14;
  const pts = mol3.at.map(a => { let x = a.x * cy + a.z * sy_, z = -a.x * sy_ + a.z * cy, y = a.y * cxr - z * sxr; z = a.y * sxr + z * cxr;
    const p = F / (F - z * 0.6); return {x: w3 / 2 + x * sc * p, y: h3 / 2 + y * sc * p, z, p, e: a.e}; });
  const items = [];
  mol3.bd.forEach(b => items.push({t: 'b', b, z: (pts[b[0]].z + pts[b[1]].z) / 2 - 0.01}));
  pts.forEach((q, i) => items.push({t: 'a', i, z: q.z}));
  items.sort((a, b) => a.z - b.z);
  items.forEach(it => {
    if (it.t === 'b') { const A = pts[it.b[0]], B = pts[it.b[1]], mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2, n = it.b[2] || 1;
      const dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) + 1e-6, ox = -dy / d, oy = dx / d, off = 3.2 * (A.p + B.p) / 2;
      for (let k = 0; k < n; k++) { const s = (k - (n - 1) / 2) * off;
        x3.lineWidth = Math.max(1.4, 3.4 * (A.p + B.p) / 2 - (n - 1)); x3.lineCap = 'round';
        x3.strokeStyle = CPK[A.e] || '#aaa'; x3.beginPath(); x3.moveTo(A.x + ox * s, A.y + oy * s); x3.lineTo(mx + ox * s, my + oy * s); x3.stroke();
        x3.strokeStyle = CPK[B.e] || '#aaa'; x3.beginPath(); x3.moveTo(mx + ox * s, my + oy * s); x3.lineTo(B.x + ox * s, B.y + oy * s); x3.stroke(); } }
    else { const q = pts[it.i], r = (RAD[q.e] || .35) * sc * q.p * 0.8, g = x3.createRadialGradient(q.x - r * .35, q.y - r * .35, r * .1, q.x, q.y, r);
      g.addColorStop(0, '#fff'); g.addColorStop(0.25, CPK[q.e] || '#aaa'); g.addColorStop(1, '#10151c');
      x3.fillStyle = g; x3.beginPath(); x3.arc(q.x, q.y, r, 0, 7); x3.fill(); } });
}
c3.addEventListener('mousedown', ev => { drag3 = [ev.clientX, ev.clientY]; });
window.addEventListener('mouseup', () => { drag3 = null; });
window.addEventListener('mousemove', ev => { if (!drag3) return; rotY += (ev.clientX - drag3[0]) * 0.012; rotX += (ev.clientY - drag3[1]) * 0.012; drag3 = [ev.clientX, ev.clientY]; });
(function l3(){ draw3(); requestAnimationFrame(l3); })();

/* ---------- selecao ---------- */
function select(i){
  sel = i; const m = M[i];
  $('ph').textContent = 'Molecule ' + (i + 1) + (m.cl < 0 ? ', no cluster' : ', cluster ' + (m.cl + 1));
  $('d2').innerHTML = '<span class="lab">2D structure</span>' + m.svg; anim2d();
  $('sq').textContent = m.qed.toFixed(2); $('sl').textContent = m.logp.toFixed(1); $('sm').textContent = Math.round(m.mw);
  typeSmi(m.smi); mol3 = m.mb ? parseMB(m.mb) : null; rotY = 0;
  const nb = $('nb'); nb.innerHTML = '';
  nbrs[i].slice(0, 4).forEach(n => { const b = document.createElement('button'); b.innerHTML = 'similar ' + (n[1] * 100).toFixed(0) + '% <b>#' + (n[0] + 1) + '</b>'; b.onclick = () => select(n[0]); nb.appendChild(b); });
}
$('replay').onclick = () => { if (sel >= 0) { $('d2').innerHTML = '<span class="lab">2D structure</span>' + M[sel].svg; anim2d(); typeSmi(M[sel].smi); } t0 = performance.now(); alpha = 0.6; };
window.__select = select;
</script></body></html>"""

def _svg(m):
    d = rdMolDraw2D.MolDraw2DSVG(320, 240)
    rdMolDraw2D.SetDarkMode(d)
    d.drawOptions().clearBackground = False
    d.drawOptions().bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return d.GetDrawingText()

def _molblock3d(m, seed=7):
    mh = Chem.AddHs(m)
    ps = AllChem.ETKDGv3(); ps.randomSeed = seed
    if AllChem.EmbedMolecule(mh, ps) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mh, maxIters=500)
    except Exception:
        pass
    return Chem.MolToMolBlock(Chem.RemoveHs(mh))

def make_viz(smiles, path="molgpt_viz.html", title="Molecules invented by the GPT", max_n=150, thr_edge=0.4):
    smiles = list(dict.fromkeys(smiles))[:max_n]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    n = len(mols)
    fps = [morgan.GetFingerprint(m) for m in mols]
    S = np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])
    dists = []
    for i in range(1, n):
        dists.extend(1 - S[i, :i])
    clusters = Butina.ClusterData(dists, n, 0.75, isDistData=True)
    cid = [-1] * n   # minus one means a singleton molecule
    for k, c in enumerate([c for c in sorted(clusters, key=len, reverse=True) if len(c) > 1]):
        for j in c:
            cid[j] = k
    edges = set()
    for i in range(n):
        k = 0
        for j in np.argsort(-S[i]):
            if j == i:
                continue
            if S[i, j] < (0.25 if k < 2 else thr_edge):
                break
            edges.add((min(i, int(j)), max(i, int(j)))); k += 1
    out = []
    for s, m, c in zip(smiles, mols, cid):
        p = props(m)
        out.append(dict(smi=s, svg=_svg(m), mb=_molblock3d(m), cl=int(c), qed=float(p["qed"]),
                        logp=float(p["logp"]), mw=float(p["mw"])))
    payload = dict(title=title, mols=out, edges=[[a, b, round(float(S[a, b]), 3)] for a, b in sorted(edges)])
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = VIZ_HTML.replace("__DATA__", blob)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{n} molecules, {len(edges)} links, {len(set(cid) - {-1})} clusters and {cid.count(-1)} singletons. File {path} ({len(html)/1e6:.1f} MB)")
    return html

# ===== 10_mostrar.py =====
viz_html = make_viz(candidates, "molgpt_viz.html", "Molecules invented by the GPT")
try:
    from IPython.display import HTML, display
    display(HTML(viz_html))
except Exception as e:
    print("open molgpt_viz.html in your browser", e)
