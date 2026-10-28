/*
 * Parser de SMILES e layout 2D, em JavaScript puro.
 *
 * O RDKit nao roda no navegador sem WASM externo, entao a estrutura precisa
 * ser montada aqui. Sao duas partes.
 *
 * 1. Parser. Subconjunto organico, atomos entre colchetes, ligacoes, ramos
 *    e fechamentos de anel. Devolve o grafo.
 * 2. Layout. Posicao inicial por caminhada em profundidade com angulo de
 *    120 graus, depois relaxamento com mola nas ligacoes e repulsao entre
 *    nao vizinhos. Um ciclo de molas iguais com repulsao relaxa sozinho
 *    para um poligono regular, entao anel nao precisa de tratamento especial.
 */

const ORG = ["Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I"];
const AROM = ["c", "n", "o", "s", "p", "b"];

export function parseSmiles(s) {
  const atoms = [], bonds = [];
  const pilha = [];
  const aneis = {};
  let anterior = -1, ordemPend = 0, i = 0;
  const parents = [];

  const addAtom = (el, arom, carga, h) => {
    atoms.push({ el, arom, carga, h, idx: atoms.length });
    return atoms.length - 1;
  };
  const addBond = (a, b, o) => {
    if (a < 0 || b < 0 || a === b) return;
    if (bonds.some(x => (x.a === a && x.b === b) || (x.a === b && x.b === a))) return;
    bonds.push({ a, b, o });
  };

  while (i < s.length) {
    const c = s[i];

    if (c === "(") { pilha.push(anterior); i++; continue; }
    if (c === ")") { anterior = pilha.pop() ?? -1; i++; continue; }
    if (c === "-") { ordemPend = 1; i++; continue; }
    if (c === "=") { ordemPend = 2; i++; continue; }
    if (c === "#") { ordemPend = 3; i++; continue; }
    if (c === ":") { ordemPend = 4; i++; continue; }
    if (c === "/" || c === "\\") { i++; continue; }
    if (c === ".") { anterior = -1; i++; continue; }

    if (c === "%" || /[0-9]/.test(c)) {
      let num;
      if (c === "%") { num = s.slice(i + 1, i + 3); i += 3; }
      else { num = c; i++; }
      if (aneis[num] !== undefined) {
        const at = aneis[num].at;
        let o = Math.max(aneis[num].o, ordemPend);
        if (!o) o = (at >= 0 && anterior >= 0 && atoms[at].arom && atoms[anterior].arom) ? 4 : 1;
        addBond(at, anterior, o);
        delete aneis[num];
      } else {
        aneis[num] = { at: anterior, o: ordemPend };
      }
      ordemPend = 0;
      continue;
    }

    let el = null, arom = false, carga = 0, h = -1;
    if (c === "[") {
      const fim = s.indexOf("]", i);
      if (fim < 0) break;
      const dentro = s.slice(i + 1, fim);
      const m = dentro.match(/^\d*([A-Z][a-z]?|[a-z])/);
      el = m ? m[1] : "C";
      arom = /^[a-z]$/.test(el);
      const mc = dentro.match(/([+-])(\d*)/);
      if (mc) carga = (mc[1] === "+" ? 1 : -1) * (mc[2] ? parseInt(mc[2]) : 1);
      const mh = dentro.match(/H(\d*)/);
      if (mh) h = mh[1] ? parseInt(mh[1]) : 1;
      i = fim + 1;
    } else {
      const dois = s.slice(i, i + 2);
      if (ORG.includes(dois)) { el = dois; i += 2; }
      else if (ORG.includes(c)) { el = c; i += 1; }
      else if (AROM.includes(c)) { el = c.toUpperCase(); arom = true; i += 1; }
      else { i += 1; continue; }
    }

    const novo = addAtom(arom ? el.toUpperCase() : el, arom, carga, h);
    parents[novo] = anterior;
    if (anterior >= 0) {
      let o = ordemPend;
      if (!o) o = (arom && atoms[anterior].arom) ? 4 : 1;
      addBond(anterior, novo, o);
    }
    ordemPend = 0;
    anterior = novo;
  }
  return { atoms, bonds };
}

/** Aneis pequenos, por busca em largura sobre o grafo. Usado so para o circulo aromatico. */
export function aneisDe(atoms, bonds, max = 7) {
  const adj = atoms.map(() => []);
  bonds.forEach((b, k) => { adj[b.a].push([b.b, k]); adj[b.b].push([b.a, k]); });
  const vistos = new Set(), aneis = [];
  for (let ini = 0; ini < atoms.length; ini++) {
    const fila = [[ini, [ini], new Set()]];
    while (fila.length) {
      const [at, cam, usadas] = fila.shift();
      if (cam.length > max) continue;
      for (const [viz, k] of adj[at]) {
        if (usadas.has(k)) continue;
        if (viz === ini && cam.length >= 3) {
          const chave = [...cam].sort((x, y) => x - y).join(",");
          if (!vistos.has(chave)) { vistos.add(chave); aneis.push([...cam]); }
          continue;
        }
        if (cam.includes(viz)) continue;
        const u = new Set(usadas); u.add(k);
        fila.push([viz, [...cam, viz], u]);
      }
    }
  }
  return aneis.filter(r => r.length <= max);
}

export function layout(atoms, bonds, iters = 420, inicial = null) {
  const n = atoms.length;
  if (!n) return [];
  const adj = atoms.map(() => []);
  bonds.forEach(b => { adj[b.a].push(b.b); adj[b.b].push(b.a); });

  // inicial, caminhada em profundidade com giro de 120 graus
  // `inicial` permite continuar de um quadro anterior, para a molecula crescer
  // suavemente em vez de pular de posicao a cada token novo
  const pos = new Array(n).fill(null);
  if (inicial) for (let i = 0; i < Math.min(n, inicial.length); i++)
    if (inicial[i]) pos[i] = { x: inicial[i].x, y: inicial[i].y };
  const ang = new Array(n).fill(0);
  const pilha = [0];
  if (!pos[0]) pos[0] = { x: 0, y: 0 };
  ang[0] = 0;
  const visto = new Set([0]);
  while (pilha.length) {
    const a = pilha.pop();
    let k = 0;
    for (const v of adj[a]) {
      if (visto.has(v)) continue;
      visto.add(v);
      const th = ang[a] + (k % 2 === 0 ? 1 : -1) * (Math.PI / 3) * (1 + Math.floor(k / 2) * 0.5);
      if (!pos[v]) pos[v] = { x: pos[a].x + Math.cos(th), y: pos[a].y + Math.sin(th) };
      ang[v] = th;
      k++;
      pilha.push(v);
    }
  }
  for (let i = 0; i < n; i++) {
    if (!pos[i]) pos[i] = { x: Math.cos(i) * 2, y: Math.sin(i) * 2 };
  }

  // relaxamento
  for (let it = 0; it < iters; it++) {
    const fx = new Float64Array(n), fy = new Float64Array(n);
    for (const b of bonds) {
      const p = pos[b.a], q = pos[b.b];
      let dx = q.x - p.x, dy = q.y - p.y;
      const d = Math.hypot(dx, dy) || 1e-6;
      const f = (d - 1) * 0.5;
      dx /= d; dy /= d;
      fx[b.a] += dx * f; fy[b.a] += dy * f;
      fx[b.b] -= dx * f; fy[b.b] -= dy * f;
    }
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y;
        const d2 = dx * dx + dy * dy + 1e-6, d = Math.sqrt(d2);
        if (d > 2.6) continue;
        const f = Math.min(0.28, 0.32 / d2);
        dx /= d; dy /= d;
        fx[i] -= dx * f; fy[i] -= dy * f;
        fx[j] += dx * f; fy[j] += dy * f;
      }
    }
    const t = 0.65 * (1 - it / iters) + 0.08;
    for (let i = 0; i < n; i++) { pos[i].x += fx[i] * t; pos[i].y += fy[i] * t; }
  }

  // normaliza, comprimento mediano de ligacao vira 1
  if (bonds.length && !inicial) {
    const ds = bonds.map(b => Math.hypot(pos[b.a].x - pos[b.b].x, pos[b.a].y - pos[b.b].y))
      .sort((a, b) => a - b);
    const med = ds[Math.floor(ds.length / 2)] || 1;
    for (const p of pos) { p.x /= med; p.y /= med; }
  }
  if (!inicial) {
    let cx = 0, cy = 0;
    for (const p of pos) { cx += p.x; cy += p.y; }
    cx /= n; cy /= n;
    for (const p of pos) { p.x -= cx; p.y -= cy; }
  }
  return pos;
}

export function qualidade(atoms, bonds, pos) {
  const lig = bonds.map(b => Math.hypot(pos[b.a].x - pos[b.b].x, pos[b.a].y - pos[b.b].y));
  const mu = lig.reduce((a, b) => a + b, 0) / Math.max(1, lig.length);
  const dp = Math.sqrt(lig.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(1, lig.length));
  const liga = new Set(bonds.map(b => `${Math.min(b.a, b.b)},${Math.max(b.a, b.b)}`));
  let minNaoLig = Infinity;
  for (let i = 0; i < atoms.length; i++)
    for (let j = i + 1; j < atoms.length; j++)
      if (!liga.has(`${i},${j}`))
        minNaoLig = Math.min(minNaoLig, Math.hypot(pos[i].x - pos[j].x, pos[i].y - pos[j].y));
  return { ligacao_media: mu, ligacao_dp: dp, min_nao_ligados: minNaoLig };
}

const MASSA = { H:1.008, B:10.81, C:12.011, N:14.007, O:15.999, F:18.998, P:30.974,
  S:32.06, Cl:35.45, Br:79.904, I:126.90 };
const VAL = { B:3, C:4, N:3, O:2, P:3, S:2, F:1, Cl:1, Br:1, I:1 };

/** Hidrogenios implicitos pela valencia padrao do subconjunto organico. */
export function hidrogenios(atoms, bonds) {
  const soma = atoms.map(() => 0), arom = atoms.map(() => false);
  for (const b of bonds) {
    const o = b.o === 4 ? 1 : b.o;
    soma[b.a] += o; soma[b.b] += o;
    if (b.o === 4) { arom[b.a] = true; arom[b.b] = true; }
  }
  return atoms.map((a, i) => {
    if (a.h >= 0) return a.h;
    let v = VAL[a.el];
    if (v === undefined) return 0;
    if (a.el === "S" && soma[i] > 2) v = soma[i];
    if (a.el === "P" && soma[i] > 3) v = 5;
    if (a.el === "N" && a.carga > 0) v = 4;
    if (a.el === "O" && a.carga < 0) v = 1;
    if (a.el === "N" && a.carga < 0) v = 2;
    // o atomo aromatico participa de uma dupla na forma de Kekule, mas so
    // quando sobra valencia para ela. N de pirrol substituido nao tem dupla.
    const bonus = arom[i] && soma[i] + 1 <= v ? 1 : 0;
    return Math.max(0, v - soma[i] - bonus);
  });
}

export function propriedades(atoms, bonds) {
  const hs = hidrogenios(atoms, bonds);
  const cont = {};
  atoms.forEach(a => { cont[a.el] = (cont[a.el] || 0) + 1; });
  const nH = hs.reduce((a, b) => a + b, 0);
  if (nH) cont.H = (cont.H || 0) + nH;
  let mw = 0;
  for (const [e, n] of Object.entries(cont)) mw += (MASSA[e] || 0) * n;
  const ordem = ["C", "H", ...Object.keys(cont).filter(e => e !== "C" && e !== "H").sort()];
  const formula = ordem.filter(e => cont[e]).map(e => e + (cont[e] > 1 ? cont[e] : "")).join("");
  const aneis = bonds.length - atoms.length + 1;
  const hbd = atoms.filter((a, i) => (a.el === "N" || a.el === "O") && hs[i] > 0).length;
  const hba = atoms.filter(a => a.el === "N" || a.el === "O").length;
  return {
    formula, mw: Math.round(mw * 100) / 100, pesados: atoms.length,
    ligacoes: bonds.length, aneis: Math.max(0, aneis), hbd, hba,
    carga: atoms.reduce((s, a) => s + a.carga, 0),
    aromaticos: atoms.filter(a => a.arom).length,
  };
}

/*
 * Conferencia de validade feita no navegador.
 *
 * Nao substitui o RDKit. Pega as quatro classes de falha que eu medi no
 * modelo treinado, por ordem de frequencia: anel sem fechar, parenteses
 * sem fechar, valencia impossivel e atomo aromatico fora de anel. A
 * kekulizacao, que e a maior classe de falha, so da para conferir com
 * percepcao de aromaticidade de verdade, entao a taxa daqui fica ACIMA
 * da taxa medida com RDKit. As duas aparecem na tela separadas.
 */
const VMAX = { B: 3, C: 4, N: 3, O: 2, P: 5, S: 6, F: 1, Cl: 1, Br: 1, I: 1 };

export function valida(s) {
  if (!s || s.length < 2) return { ok: false, motivo: "vazio" };
  let prof = 0;
  for (const c of s) {
    if (c === "(") prof++;
    else if (c === ")") { prof--; if (prof < 0) return { ok: false, motivo: "parenteses" }; }
  }
  if (prof !== 0) return { ok: false, motivo: "parenteses" };

  // fechamentos de anel pendentes
  const abertos = {};
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === "[") { const f = s.indexOf("]", i); if (f < 0) return { ok: false, motivo: "colchete" }; i = f + 1; continue; }
    if (c === "%") { const num = s.slice(i + 1, i + 3); abertos[num] = !abertos[num]; i += 3; continue; }
    if (/[0-9]/.test(c)) { abertos[c] = !abertos[c]; i += 1; continue; }
    i += 1;
  }
  if (Object.values(abertos).some(v => v)) return { ok: false, motivo: "anel sem fechar" };

  const { atoms, bonds } = parseSmiles(s);
  if (atoms.length < 2) return { ok: false, motivo: "atomos de menos" };

  const soma = atoms.map(() => 0), arom = atoms.map(() => false);
  for (const b of bonds) {
    const o = b.o === 4 ? 1 : b.o;
    soma[b.a] += o; soma[b.b] += o;
    if (b.o === 4) { arom[b.a] = true; arom[b.b] = true; }
  }
  for (let k = 0; k < atoms.length; k++) {
    const a = atoms[k];
    let v = VMAX[a.el];
    if (v === undefined) continue;
    v += a.carga > 0 && (a.el === "N" || a.el === "P") ? 1 : 0;
    v -= a.carga < 0 && (a.el === "O" || a.el === "N" || a.el === "S") ? 1 : 0;
    const usado = soma[k] + (a.h > 0 ? a.h : 0);
    if (usado > v) return { ok: false, motivo: "valencia" };
  }
  const nArom = atoms.map(() => 0);
  for (const b of bonds) if (b.o === 4) { nArom[b.a]++; nArom[b.b]++; }
  for (let k = 0; k < atoms.length; k++)
    if (atoms[k].arom && nArom[k] < 2) return { ok: false, motivo: "aromatico fora de anel" };

  // fragmento solto de um atomo
  const grau = atoms.map(() => 0);
  for (const b of bonds) { grau[b.a]++; grau[b.b]++; }
  if (grau.some(g => g === 0)) return { ok: false, motivo: "atomo solto" };
  return { ok: true, motivo: "aceita" };
}
