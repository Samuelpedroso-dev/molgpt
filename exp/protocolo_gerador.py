"""
Protocolo fixo do gerador. Cinco temperaturas, cinco mil moleculas em cada,
semente fixa.

Mede o que se mede em benchmark de modelo generativo molecular.

  validade      o RDKit aceita a molecula
  unicidade     fracao de SMILES canonicos distintos entre as validas
  novidade      fracao das validas que NAO esta nas 249 mil do treino
  diversidade   1 menos a Tanimoto media entre pares, nas validas

E mede tambem a conferencia do navegador contra a do RDKit, para quantificar
o quanto a conferencia barata aceita a mais.
"""
import json, os, sys, time, random
import numpy as np
import torch

sys.path.insert(0, "/home/claude/treino")
sys.path.insert(0, "/home/claude/quim")
from treinar import MolGPT  # noqa: E402
from quimica import valida as valida_navegador  # noqa: E402
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(2)

N = int(os.environ.get("N_AMOSTRAS", 5000))
LOTE = 250
TEMPS = (0.6, 0.7, 0.8, 0.9, 1.0)


def canonico(s):
    m = Chem.MolFromSmiles(s)
    return None if m is None else Chem.MolToSmiles(m)


@torch.no_grad()
def amostra(m, cfg, itos, stoi, n, temp, semente):
    """Amostragem em lote. Cada passo processa a sequencia inteira, sem cache,
    porque aqui o que importa e o numero, nao a latencia."""
    torch.manual_seed(semente)
    saidas = []
    for ini in range(0, n, LOTE):
        b = min(LOTE, n - ini)
        faixa = torch.randint(0, 5, (b,))
        seq = torch.stack([torch.tensor([1, stoi[f"<lp{int(f)}>"]]) for f in faixa])
        vivo = torch.ones(b, dtype=torch.bool)
        txt = [[] for _ in range(b)]
        for _ in range(110):
            lg = m(seq)[0][:, -1, :] / temp
            lg[:, 0] = -1e9; lg[:, 1] = -1e9
            p = torch.softmax(lg, -1)
            nxt = torch.multinomial(p, 1).squeeze(-1)
            for k in range(b):
                if not vivo[k]:
                    continue
                if nxt[k].item() == 2:
                    vivo[k] = False
                else:
                    t = itos[nxt[k].item()]
                    if not t.startswith("<"):
                        txt[k].append(t)
            if not vivo.any():
                break
            seq = torch.cat([seq, nxt[:, None]], 1)
            if seq.shape[1] >= cfg["block_size"]:
                break
        saidas += ["".join(t) for t in txt]
    return saidas[:n]


def main():
    CK = os.environ.get("CKPT", "/home/claude/treino/saida/ckpt.pt")
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    print(f"checkpoint {CK}, passo {ck['iter']}")
    cfg, itos = ck["cfg"], ck["itos"]
    stoi = {t: i for i, t in enumerate(itos)}
    m = MolGPT(len(itos), cfg["block_size"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"], 0.0)
    m.load_state_dict(ck["model"]); m.eval()

    treino = set()
    import csv
    for l in csv.DictReader(open("/home/claude/treino/zinc250k.csv")):
        treino.add(l["smiles"].strip())
    treino_can = set()
    print(f"conjunto de treino, {len(treino)} SMILES. canonizando para a novidade...", flush=True)
    for s in treino:
        c = canonico(s)
        if c:
            treino_can.add(c)
    print(f"  {len(treino_can)} canonicos\n", flush=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    res = []
    for temp in TEMPS:
        t0 = time.time()
        sm = amostra(m, cfg, itos, stoi, N, temp, semente=7)
        cans, motivos, nav_ok, acordo = [], {}, 0, {"ambos": 0, "so navegador": 0, "so rdkit": 0,
                                                    "nenhum": 0}
        for s in sm:
            c = canonico(s)
            nv = valida_navegador(s)[0]
            nav_ok += nv
            if c:
                cans.append(c)
            else:
                mot = valida_navegador(s)[1]
                motivos[mot] = motivos.get(mot, 0) + 1
            acordo["ambos" if (c and nv) else "so navegador" if nv else
                   "so rdkit" if c else "nenhum"] += 1
        unicos = set(cans)
        novos = unicos - treino_can
        amostraM = random.Random(0).sample(sorted(unicos), min(1000, len(unicos)))
        fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in amostraM]
        sims = []
        for i in range(len(fps)):
            sims += DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        r = dict(temp=temp, n=N,
                 validade_rdkit=round(len(cans) / N, 4),
                 validade_navegador=round(nav_ok / N, 4),
                 unicidade=round(len(unicos) / max(1, len(cans)), 4),
                 novidade=round(len(novos) / max(1, len(unicos)), 4),
                 diversidade=round(1 - float(np.mean(sims)), 4),
                 acordo=acordo, motivos_navegador=motivos)
        res.append(r)
        print(f"T {temp}  validade RDKit {r['validade_rdkit']:.3f}  navegador "
              f"{r['validade_navegador']:.3f}  unicidade {r['unicidade']:.3f}  "
              f"novidade {r['novidade']:.3f}  diversidade {r['diversidade']:.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"   acordo {acordo}", flush=True)
    json.dump(res, open("/home/claude/exp/saida/protocolo_gerador.json", "w"), indent=1)
    print("\ngravado")


if __name__ == "__main__":
    main()
