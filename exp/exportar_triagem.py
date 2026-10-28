"""
Exporta os tres modelos de triagem para o navegador.

Sai triagem.bin, com todos os pesos em float16 numa ordem fixa, e
triagem.json, com o mapa de deslocamentos, as metricas medidas de cada
modelo, os 442 alvos do Davis ja codificados, os quantis de referencia
calculados em moleculas do ZINC e as impressoes digitais das 68 moleculas
do Davis, para o calculo de dominio de aplicacao no navegador.
"""
import base64, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/claude/exp")
sys.path.insert(0, "/home/claude/quim")
from quimica import digital  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

SAIDA = "/home/claude/exp/saida"
WEB = "/home/claude/web"
F = "/home/claude/exp/feat"
torch.set_num_threads(2)

tensores, mapa, off = [], {}, 0


def guarda(nome, t):
    global off
    a = t.detach().cpu().numpy().astype(np.float16)
    mapa[nome] = {"off": off, "shape": list(a.shape)}
    tensores.append(a.tobytes())
    off += a.size


def guarda_seq(prefixo, seq):
    """Guarda as camadas lineares de um nn.Sequential, na ordem."""
    k = 0
    for cam in seq:
        if isinstance(cam, nn.Linear):
            guarda(f"{prefixo}.{k}.w", cam.weight)
            guarda(f"{prefixo}.{k}.b", cam.bias)
            k += 1
    return k


def rede(nin, nout, h1=128, h2=64, drop=0.0):
    return nn.Sequential(nn.Linear(nin, h1), nn.GELU(), nn.Dropout(drop),
                         nn.Linear(h1, h2), nn.GELU(), nn.Dropout(drop), nn.Linear(h2, nout))


def main():
    meta = {"versao": 1}

    # ---------- sintetizabilidade ----------
    sas = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(),
                        nn.Linear(64, 1))
    sas.load_state_dict({k.replace("r.", ""): v for k, v in
                         torch.load(f"{SAIDA}/sas.pt", map_location="cpu").items()})
    sas.eval()
    n = guarda_seq("sas", sas)
    meta["sas"] = {"camadas": n, "nin": 1024}

    # ---------- toxicologia ----------
    tm = json.load(open(f"{SAIDA}/tox_metricas.json"))
    tox = rede(tm["nbits"], 12, tm["h1"], tm["h2"], tm["drop"])
    tox.load_state_dict(torch.load(f"{SAIDA}/tox_s0.pt", map_location="cpu"))
    tox.eval()
    n = guarda_seq("tox", tox)
    meta["tox"] = {"camadas": n, "nin": tm["nbits"], "ensaios": tm["ensaios"]}

    # AUC da semente 0 sozinha, que e a que vai para o navegador
    X = np.load(f"{F}/tox_X.npy"); Y = np.load(f"{F}/tox_Y.npy")
    sm = open(f"{F}/tox_smiles.txt").read().split("\n")
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    grupos = {}
    for i, s in enumerate(sm):
        m_ = Chem.MolFromSmiles(s)
        k = MurckoScaffold.MurckoScaffoldSmiles(mol=m_, includeChirality=False) if m_ else s
        grupos.setdefault(k, []).append(i)
    tr, va, te = [], [], []
    for g in sorted(grupos.values(), key=len, reverse=True):
        if len(tr) < len(sm) * .7: tr.extend(g)
        elif len(va) < len(sm) * .1: va.extend(g)
        else: te.extend(g)
    with torch.no_grad():
        lg = tox(torch.tensor(X[te])).numpy()
    auc0 = {}
    for j, e in enumerate(tm["ensaios"]):
        ok = ~np.isnan(Y[te][:, j])
        if ok.sum() > 20 and 0 < Y[te][ok, j].sum() < ok.sum():
            auc0[e] = round(float(roc_auc_score(Y[te][ok, j], lg[ok, j])), 4)
    meta["tox"]["auc_desta_semente"] = auc0
    meta["tox"]["auc_3_sementes"] = tm["auc"]
    meta["tox"]["auc_media"] = tm["media"]
    meta["tox"]["n_teste"] = tm["n_teste"]

    # ---------- afinidade ----------
    ck = torch.load(f"{SAIDA}/afinidade.pt", map_location="cpu", weights_only=False)  # arquivo nosso, gravado nesta sessao
    sd, V = ck["model"], ck["V"]
    f = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 64))
    f.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("f.")})
    h = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(),
                      nn.Linear(64, 1))
    h.load_state_dict({k.replace("h.0", "0").replace("h.3", "2").replace("h.5", "4"): v
                       for k, v in sd.items() if k.startswith("h.")})
    f.eval(); h.eval()
    nf = guarda_seq("af_f", f)
    nh = guarda_seq("af_h", h)
    guarda("af_V", torch.tensor(V))
    am = json.load(open(f"{SAIDA}/afinidade_metricas.json"))
    meta["afinidade"] = {"camadas_f": nf, "camadas_h": nh, "nin": 1024,
                         "proteinas": am["proteinas"], "metricas": am["metricas"],
                         "piso": am["piso"], "faixa_pkd": am["faixa_pkd"]}
    # a tabela completa vai para o HTML na montagem, nao para o json do navegador,
    # porque ela tem NaN nas linhas de baseline constante e isso nao e JSON valido

    # ---------- referencias calculadas em moleculas do ZINC ----------
    Xz = np.load(f"{F}/zinc_X.npy")
    smz = open(f"{F}/zinc_smiles.txt").read().split("\n")
    rng = np.random.RandomState(0)
    idx = rng.choice(len(Xz), 2000, replace=False)
    fp1024 = torch.tensor(Xz[idx])
    with torch.no_grad():
        U = f(fp1024)                                    # 2000 x 64
        Vt = torch.tensor(V)                             # 442 x 64
        quantis = np.linspace(0, 100, 21)
        qa = np.zeros((Vt.shape[0], 21), dtype=np.float32)
        for j in range(Vt.shape[0]):
            v = Vt[j][None, :].expand(U.shape[0], -1)
            p = h(torch.cat([U, v, U * v], -1)).squeeze(-1).numpy()
            qa[j] = np.percentile(p, quantis)
        fp_tox = torch.tensor(np.stack([digital(smz[i], tm["nbits"]) for i in idx])
                              .astype(np.float32))
        lt = tox(fp_tox).numpy()
        qt = np.percentile(lt, quantis, axis=0).T.astype(np.float32)   # 12 x 21
        ps = sas(fp1024).squeeze(-1).numpy()
    guarda("q_afinidade", torch.tensor(qa))
    guarda("q_tox", torch.tensor(qt))
    meta["quantis"] = list(quantis)
    meta["sas"]["zinc"] = {"mediana": float(np.median(ps)),
                           "q10": float(np.percentile(ps, 10)),
                           "q90": float(np.percentile(ps, 90)), "n": 2000}
    tmet = json.load(open(f"{SAIDA}/triagem_metricas.json"))
    meta["sas"]["metricas"] = tmet["sas"]

    # ---------- dominio de aplicacao, as 68 do Davis empacotadas em bits ----------
    bits = am["davis_bits"]
    emp = np.zeros((len(bits), 1024 // 8), dtype=np.uint8)
    for i, b in enumerate(bits):
        for k in b:
            emp[i, k // 8] |= 1 << (k % 8)
    meta["davis_bits_b64"] = base64.b64encode(emp.tobytes()).decode()
    meta["davis_n"] = len(bits)

    bin_path = f"{WEB}/triagem.bin"
    with open(bin_path, "wb") as fh:
        for t in tensores:
            fh.write(t)
    with open(f"{WEB}/triagem.b64.txt", "w") as fh:
        fh.write(base64.b64encode(open(bin_path, "rb").read()).decode())
    meta["mapa"] = mapa
    meta["n_valores"] = int(off)
    json.dump(meta, open(f"{WEB}/triagem.json", "w"), separators=(",", ":"), allow_nan=False)

    # ---------- caso de referencia para o JavaScript conferir ----------
    s = "CC(=O)Oc1ccccc1C(=O)O"
    fp = torch.tensor(np.array(digital(s, 1024), dtype=np.float32))[None, :]
    fpt = torch.tensor(np.array(digital(s, tm["nbits"]), dtype=np.float32))[None, :]
    with torch.no_grad():
        u = f(fp); v = Vt[0][None, :]
        ref = {"smiles": s,
               "sas": round(float(sas(fp)[0, 0]), 5),
               "tox": [round(float(x), 5) for x in tox(fpt)[0]],
               "afinidade_proteina_0": round(float(h(torch.cat([u, v, u * v], -1))[0, 0]), 5)}
    json.dump(ref, open(f"{WEB}/triagem_referencia.json", "w"), indent=1)
    print(f"triagem.bin {os.path.getsize(bin_path)/1e6:.2f} MB, {off:,} valores")
    print(f"triagem.json {os.path.getsize(f'{WEB}/triagem.json')/1024:.0f} KB")
    print("referencia", json.dumps(ref)[:160])


if __name__ == "__main__":
    main()
