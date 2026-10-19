"""
Por que a validade estagnou perto de 57 por cento enquanto a perda cai.

Tres hipoteses testadas aqui.

H1. Temperatura. Amostrar a 1.0 inclui tokens de baixa probabilidade que quebram
    a sintaxe. Testar uma curva de temperatura.
H2. Truncagem. Se o modelo nao emite o token de fim, a sequencia e cortada e vira
    invalida por acidente, nao por erro de quimica.
H3. Desalinhamento de posicao. No treino empacotado o token de inicio aparece em
    posicao qualquer da janela, mas na geracao ele esta sempre na posicao zero.
    A embedding de posicao zero quase nunca viu um inicio de molecula.

Tambem classifica as invalidas por tipo de falha, que diz o que o modelo ainda
nao aprendeu.
"""
import os, re, sys, json
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(1)          # o treino esta usando as 2 CPUs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from treinar import MolGPT, CFG   # noqa: E402

SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saida")
ck = torch.load(os.path.join(SAIDA, "ckpt.pt"), map_location="cpu", weights_only=False)
itos = ck["itos"]
stoi = {t: i for i, t in enumerate(itos)}
BOS, EOS, PAD = 1, 2, 0
ctx = CFG["block_size"]

modelo = MolGPT(len(itos), ctx, CFG["n_embd"], CFG["n_head"], CFG["n_layer"], 0.0)
modelo.load_state_dict(ck["model"])
modelo.eval()
print(f"checkpoint do passo {ck['iter']}")


@torch.no_grad()
def gerar(n, temp=1.0, top_k=None, max_novos=110, prefixo_pos=0):
    ini = [PAD] * prefixo_pos + [BOS]
    idx = torch.tensor([ini] * n, dtype=torch.long)
    pronto = torch.zeros(n, dtype=torch.bool)
    truncadas = torch.ones(n, dtype=torch.bool)
    for _ in range(max_novos):
        logits = modelo(idx[:, -ctx:])[0][:, -1].float() / temp
        logits[:, PAD] = -1e9
        logits[:, BOS] = -1e9
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -1e9
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        nxt[pronto] = PAD
        idx = torch.cat([idx, nxt], 1)
        novo_fim = (nxt.squeeze(1) == EOS) & (~pronto)
        truncadas[novo_fim] = False
        pronto |= nxt.squeeze(1) == EOS
        if pronto.all():
            break
    saida = []
    for linha, trunc in zip(idx.tolist(), truncadas.tolist()):
        t = []
        for x in linha[prefixo_pos + 1:]:
            if x in (EOS, PAD):
                break
            if itos[x].startswith("<"):
                continue
            t.append(itos[x])
        saida.append(("".join(t), trunc))
    return saida


def classificar(s):
    """Tipo de falha de um SMILES invalido."""
    if not s:
        return "vazia"
    if s.count("(") != s.count(")"):
        return "parenteses desbalanceados"
    digitos = re.findall(r"(?<!\[)(?<!%)[1-9]", s)
    impares = [d for d in set(digitos) if digitos.count(d) % 2 == 1]
    if impares:
        return "anel nao fechado"
    try:
        m = Chem.MolFromSmiles(s, sanitize=False)
        if m is None:
            return "sintaxe, outra"
        Chem.SanitizeMol(m)
        return "valida apos sanitizar"
    except Exception as e:
        t = str(e)
        if "Kekulize" in t or "kekul" in t.lower():
            return "kekulizacao aromatica"
        if "valence" in t.lower():
            return "valencia"
        return "sanitizacao, outra"


def avaliar(amostras, rotulo):
    n = len(amostras)
    validas, falhas, trunc_inval = 0, {}, 0
    for s, trunc in amostras:
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None and m.GetNumAtoms() > 0:
            validas += 1
            continue
        if trunc:
            trunc_inval += 1
            k = "truncada, sem token de fim"
        else:
            k = classificar(s)
        falhas[k] = falhas.get(k, 0) + 1
    print(f"\n{rotulo}  validade {100*validas/n:.1f}%  ({validas}/{n})")
    for k, v in sorted(falhas.items(), key=lambda kv: -kv[1]):
        print(f"    {k:32s} {v:4d}  ({100*v/max(1,n-validas):.0f}% das invalidas)")
    return validas / n


N = 400
print("\n=== H1, curva de temperatura ===")
res_t = {}
for t in (1.0, 0.9, 0.8, 0.7, 0.6):
    res_t[t] = avaliar(gerar(N, temp=t), f"temperatura {t}")

print("\n=== H1b, top k a temperatura 1.0 ===")
for k in (10, 20):
    avaliar(gerar(N, temp=1.0, top_k=k), f"top k {k}")

print("\n=== H3, token de inicio longe da posicao zero ===")
for p in (0, 8, 24):
    avaliar(gerar(N, temp=0.8, prefixo_pos=p), f"inicio na posicao {p}")

with open(os.path.join(SAIDA, "diagnostico.json"), "w") as f:
    json.dump({"iter": ck["iter"], "validade_por_temperatura": res_t}, f, indent=1)
print("\ngravado em saida/diagnostico.json")
