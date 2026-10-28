"""Modelo de afinidade que vai para o navegador, treinado em todos os pares.
As metricas honestas estao em tabela_afinidade.json, medidas em split."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/home/claude/exp"); sys.path.insert(0, "/home/claude/quim")
sys.path.insert(0, "/home/claude/pkg")
from tabela_afinidade import Interativo, treina_rede  # noqa: E402
from teto_proteico import carregar, desc400  # noqa: E402
from quimica import digital  # noqa: E402
torch.set_num_threads(2)

smiles, seqs, pKd = carregar()
nd, npr = len(smiles), len(seqs)
nomes = list(json.load(open("/home/claude/dados_davis/proteins.txt")).keys())
FP = np.stack([np.array(digital(s, 1024), dtype=np.float32) for s in smiles])
Xd = torch.tensor(FP); Xp = torch.tensor(np.stack([desc400(s) for s in seqs]))
pares = [(i, j) for i in range(nd) for j in range(npr)]
y = np.array([pKd[i, j] for i, j in pares], dtype=np.float32)
todos = list(range(len(pares)))
m, _ = treina_rede(Interativo, Xd, Xp, y, todos, todos[:2], pares, 0)
with torch.no_grad():
    V = m.g(Xp).numpy()
torch.save({"model": m.state_dict(), "V": V}, "/home/claude/exp/saida/afinidade.pt")
tab = json.load(open("/home/claude/exp/saida/tabela_afinidade.json"))
json.dump({"metricas": {r["split"]: r for r in tab if r["modelo"] == "interativo"},
           "n_proteinas": npr, "n_moleculas": nd,
           "piso": float(np.isclose(y, 5.0).mean()), "proteinas": nomes,
           "davis_bits": [sorted(np.nonzero(FP[i])[0].tolist()) for i in range(nd)],
           "faixa_pkd": [float(y.min()), float(y.max())]},
          open("/home/claude/exp/saida/afinidade_metricas.json", "w"))
print("afinidade.pt gravado, V", V.shape)
