"""
Exporta o checkpoint para um formato que roda no navegador.

Saem dois arquivos.
  molgpt.bin   todos os pesos em float16, concatenados numa ordem fixa
  molgpt.json  vocabulario, configuracao e o mapa de nome para deslocamento

O modelo tem 819.328 parametros unicos. Em float16 sao 1,6 MB, pequeno o
bastante para o navegador baixar uma vez e gerar em tempo real.

A cabeca de saida compartilha peso com a embedding de token, entao ela nao
e exportada, o JavaScript reusa a mesma matriz.
"""
import json, os, struct
import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
CK = os.environ.get("CKPT", "/home/claude/treino/saida/ckpt.pt")
PROT = "/home/claude/exp/saida/protocolo_gerador.json"

ck = torch.load(CK, map_location="cpu", weights_only=False)
sd, cfg, itos = ck["model"], ck["cfg"], ck["itos"]
L, d, H = cfg["n_layer"], cfg["n_embd"], cfg["n_head"]

ordem = ["tok.weight", "pos.weight"]
for i in range(L):
    ordem += [f"blocos.{i}.ln1.weight", f"blocos.{i}.ln1.bias",
              f"blocos.{i}.qkv.weight", f"blocos.{i}.qkv.bias",
              f"blocos.{i}.proj.weight", f"blocos.{i}.proj.bias",
              f"blocos.{i}.ln2.weight", f"blocos.{i}.ln2.bias",
              f"blocos.{i}.mlp.0.weight", f"blocos.{i}.mlp.0.bias",
              f"blocos.{i}.mlp.2.weight", f"blocos.{i}.mlp.2.bias"]
ordem += ["ln_f.weight", "ln_f.bias"]

faltando = [k for k in ordem if k not in sd]
assert not faltando, f"tensores ausentes, {faltando}"

mapa, blobs, off = {}, [], 0
for nome in ordem:
    t = sd[nome].detach().cpu().numpy().astype(np.float16)
    mapa[nome] = {"off": off, "shape": list(t.shape)}
    blobs.append(t.tobytes())
    off += t.size

bin_path = os.path.join(AQUI, "molgpt.bin")
with open(bin_path, "wb") as f:
    for b in blobs:
        f.write(b)

meta = {
    "arquitetura": "molgpt",
    "n_layer": L, "n_head": H, "n_embd": d,
    "block_size": cfg["block_size"],
    "vocab": itos,
    "pad": 0, "bos": 1, "eos": 2,
    "n_params": int(off),
    "dtype": "float16",
    "mapa": mapa,
    "passo_treino": int(ck["iter"]),
    # validade medida pelo protocolo fixo, no proprio checkpoint que vai junto
    "validade_medida": {f"{r['temp']:.1f}": r["validade_rdkit"]
                        for r in json.load(open(PROT))} if os.path.exists(PROT) else {},
    "validade_navegador": {f"{r['temp']:.1f}": r["validade_navegador"]
                           for r in json.load(open(PROT))} if os.path.exists(PROT) else {},
    "faixas_logp": {"0": 0.80, "1": 1.97, "2": 2.55, "3": 3.12, "4": 3.73},
}
json.dump(meta, open(os.path.join(AQUI, "molgpt.json"), "w"), separators=(",", ":"))

# o navegador le os pesos em base64, porque e o formato que tanto o GitHub
# quanto o hospedeiro da pagina servem sem combinacao previa
import base64
b64_path = os.path.join(AQUI, "molgpt.b64.txt")
with open(bin_path, "rb") as f, open(b64_path, "w") as g:
    g.write(base64.b64encode(f.read()).decode())

print(f"molgpt.bin  {os.path.getsize(bin_path)/1e6:.2f} MB, {off:,} valores")
print(f"molgpt.b64.txt {os.path.getsize(b64_path)/1e6:.2f} MB")
print(f"molgpt.json {os.path.getsize(os.path.join(AQUI,'molgpt.json'))/1024:.0f} KB, "
      f"vocabulario {len(itos)}")

# ---- caso de teste para o JavaScript conferir contra o torch ----
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/claude/treino")
from treinar import MolGPT  # noqa: E402

m = MolGPT(len(itos), cfg["block_size"], d, H, L, 0.0)
m.load_state_dict(sd); m.eval()
entrada = [1, 3, 10, 20, 21, 22, 30]          # bos mais alguns tokens
with torch.no_grad():
    lg = m(torch.tensor([entrada]))[0][0, -1]
teste = {
    "entrada": entrada,
    "logits_ultimos": [round(float(x), 5) for x in lg[:12]],
    "soma_logits": round(float(lg.sum()), 5),
    "argmax": int(lg.argmax()),
}
json.dump(teste, open(os.path.join(AQUI, "teste_referencia.json"), "w"), indent=1)
print("\ncaso de referencia gravado")
print("  entrada", entrada)
print("  argmax", teste["argmax"], "| soma dos logits", teste["soma_logits"])
print("  12 primeiros logits", teste["logits_ultimos"][:6], "...")
