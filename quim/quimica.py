"""
Porte fiel do parser de SMILES que roda no navegador, mais a impressao
digital circular usada pelos tres modelos.

A regra e simples. O modelo so vale no navegador se as features forem
IDENTICAS nos dois lados. Entao nada de RDKit aqui dentro. O RDKit entra
so para dar rotulo, nunca para dar feature.
"""
ORG = ["Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I"]
AROM = ["c", "n", "o", "s", "p", "b"]
VAL = {"B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}
VMAX = {"B": 3, "C": 4, "N": 3, "O": 2, "P": 5, "S": 6, "F": 1, "Cl": 1, "Br": 1, "I": 1}
ELS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]

import re
_RE_EL = re.compile(r"^\d*([A-Z][a-z]?|[a-z])")
_RE_CARGA = re.compile(r"([+-])(\d*)")
_RE_H = re.compile(r"H(\d*)")


def parse_smiles(s):
    atoms, bonds = [], []
    pilha, aneis = [], {}
    anterior, ordem_pend, i = -1, 0, 0

    def add_bond(a, b, o):
        if a < 0 or b < 0 or a == b:
            return
        for x in bonds:
            if (x[0] == a and x[1] == b) or (x[0] == b and x[1] == a):
                return
        bonds.append((a, b, o))

    while i < len(s):
        c = s[i]
        if c == "(":
            pilha.append(anterior); i += 1; continue
        if c == ")":
            anterior = pilha.pop() if pilha else -1; i += 1; continue
        if c == "-":
            ordem_pend = 1; i += 1; continue
        if c == "=":
            ordem_pend = 2; i += 1; continue
        if c == "#":
            ordem_pend = 3; i += 1; continue
        if c == ":":
            ordem_pend = 4; i += 1; continue
        if c in "/\\":
            i += 1; continue
        if c == ".":
            anterior = -1; i += 1; continue
        if c == "%" or c.isdigit():
            if c == "%":
                num = s[i + 1:i + 3]; i += 3
            else:
                num = c; i += 1
            if num in aneis:
                at, o0 = aneis.pop(num)
                o = max(o0, ordem_pend)
                if not o:
                    o = 4 if (at >= 0 and anterior >= 0 and atoms[at]["arom"]
                              and atoms[anterior]["arom"]) else 1
                add_bond(at, anterior, o)
            else:
                aneis[num] = (anterior, ordem_pend)
            ordem_pend = 0
            continue

        el, arom, carga, h = None, False, 0, -1
        if c == "[":
            fim = s.find("]", i)
            if fim < 0:
                break
            dentro = s[i + 1:fim]
            m = _RE_EL.match(dentro)
            el = m.group(1) if m else "C"
            arom = bool(re.fullmatch(r"[a-z]", el))
            mc = _RE_CARGA.search(dentro)
            if mc:
                carga = (1 if mc.group(1) == "+" else -1) * (int(mc.group(2)) if mc.group(2) else 1)
            mh = _RE_H.search(dentro)
            if mh:
                h = int(mh.group(1)) if mh.group(1) else 1
            i = fim + 1
        else:
            dois = s[i:i + 2]
            if dois in ORG:
                el = dois; i += 2
            elif c in ORG:
                el = c; i += 1
            elif c in AROM:
                el = c.upper(); arom = True; i += 1
            else:
                i += 1; continue

        atoms.append({"el": el.upper() if arom else el, "arom": arom, "carga": carga, "h": h})
        novo = len(atoms) - 1
        if anterior >= 0:
            o = ordem_pend
            if not o:
                o = 4 if (arom and atoms[anterior]["arom"]) else 1
            add_bond(anterior, novo, o)
        ordem_pend = 0
        anterior = novo
    return atoms, bonds


def hidrogenios(atoms, bonds):
    soma = [0] * len(atoms)
    arom = [False] * len(atoms)
    for a, b, o in bonds:
        oo = 1 if o == 4 else o
        soma[a] += oo; soma[b] += oo
        if o == 4:
            arom[a] = True; arom[b] = True
    hs = []
    for i, a in enumerate(atoms):
        if a["h"] >= 0:
            hs.append(a["h"]); continue
        v = VAL.get(a["el"])
        if v is None:
            hs.append(0); continue
        if a["el"] == "S" and soma[i] > 2:
            v = soma[i]
        if a["el"] == "P" and soma[i] > 3:
            v = 5
        if a["el"] == "N" and a["carga"] > 0:
            v = 4
        if a["el"] == "O" and a["carga"] < 0:
            v = 1
        if a["el"] == "N" and a["carga"] < 0:
            v = 2
        bonus = 1 if (arom[i] and soma[i] + 1 <= v) else 0
        hs.append(max(0, v - soma[i] - bonus))
    return hs


def valida(s):
    if not s or len(s) < 2:
        return False, "vazio"
    prof = 0
    for c in s:
        if c == "(":
            prof += 1
        elif c == ")":
            prof -= 1
            if prof < 0:
                return False, "parenteses"
    if prof:
        return False, "parenteses"
    abertos, i = {}, 0
    while i < len(s):
        c = s[i]
        if c == "[":
            f = s.find("]", i)
            if f < 0:
                return False, "colchete"
            i = f + 1; continue
        if c == "%":
            num = s[i + 1:i + 3]; abertos[num] = not abertos.get(num); i += 3; continue
        if c.isdigit():
            abertos[c] = not abertos.get(c); i += 1; continue
        i += 1
    if any(abertos.values()):
        return False, "anel sem fechar"
    atoms, bonds = parse_smiles(s)
    if len(atoms) < 2:
        return False, "atomos de menos"
    soma = [0] * len(atoms); narom = [0] * len(atoms); grau = [0] * len(atoms)
    for a, b, o in bonds:
        soma[a] += 1 if o == 4 else o
        soma[b] += 1 if o == 4 else o
        grau[a] += 1; grau[b] += 1
        if o == 4:
            narom[a] += 1; narom[b] += 1
    for k, a in enumerate(atoms):
        v = VMAX.get(a["el"])
        if v is None:
            continue
        if a["carga"] > 0 and a["el"] in ("N", "P"):
            v += 1
        if a["carga"] < 0 and a["el"] in ("O", "N", "S"):
            v -= 1
        if soma[k] + (a["h"] if a["h"] > 0 else 0) > v:
            return False, "valencia"
    for k, a in enumerate(atoms):
        if a["arom"] and narom[k] < 2:
            return False, "aromatico fora de anel"
    if any(g == 0 for g in grau):
        return False, "atomo solto"
    return True, "aceita"


# ---------- impressao digital circular ----------
M32 = 0xFFFFFFFF


def _mix(h, x):
    """Mistura de 32 bits. Precisa dar o mesmo resultado que Math.imul no JS."""
    h = (h ^ (x & M32)) & M32
    h = (h * 0x01000193) & M32          # primo do FNV de 32 bits
    h = (h ^ (h >> 15)) & M32
    return h


def _invariante(a, h_impl, grau):
    cod = ELS.index(a["el"]) + 1 if a["el"] in ELS else 20
    v = cod
    v = v * 8 + min(grau, 7)
    v = v * 8 + min(h_impl, 7)
    v = v * 8 + (a["carga"] + 3 if -3 <= a["carga"] <= 3 else 7)
    v = v * 2 + (1 if a["arom"] else 0)
    return v & M32


def digital(smiles, nbits=1024, raio=2, atoms=None, bonds=None):
    """
    ECFP na forma classica. Invariante por atomo, depois duas rodadas em que
    cada atomo absorve os vizinhos, e cada identificador que aparece vira um
    bit. Nao e a do RDKit e nem tenta ser, e a mesma dos dois lados.
    """
    if atoms is None:
        atoms, bonds = parse_smiles(smiles)
    n = len(atoms)
    bits = [0] * nbits
    if not n:
        return bits
    viz = [[] for _ in range(n)]
    for a, b, o in bonds:
        viz[a].append((b, o)); viz[b].append((a, o))
    hs = hidrogenios(atoms, bonds)
    ident = [_invariante(atoms[i], hs[i], len(viz[i])) for i in range(n)]
    for i in range(n):
        bits[ident[i] % nbits] = 1
    for _ in range(raio):
        novo = []
        for i in range(n):
            vz = sorted(((o, ident[j]) for j, o in viz[i]))
            h = _mix(2166136261, ident[i])
            for o, idj in vz:
                h = _mix(h, o)
                h = _mix(h, idj)
            novo.append(h)
        ident = novo
        for i in range(n):
            bits[ident[i] % nbits] = 1
    return bits
