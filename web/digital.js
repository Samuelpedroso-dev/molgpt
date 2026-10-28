// a impressao digital usa o mesmo grafo e os mesmos hidrogenios do parser
import { hidrogenios } from "./quimica.js";
/*
 * Impressao digital circular, a mesma do lado Python.
 *
 * Os tres modelos de triagem foram treinados com ESTES bits. Se a conta aqui
 * divergir um bit da conta do treino, a previsao vira ruido, entao a
 * aritmetica e de 32 bits dos dois lados e existe um teste que compara os
 * dois em centenas de moleculas.
 */
const ELS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"];

function mix32(h, x) {
  h = (h ^ (x >>> 0)) >>> 0;
  h = Math.imul(h, 0x01000193) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  return h;
}

function invariante(a, hImpl, grau) {
  const cod = ELS.indexOf(a.el) >= 0 ? ELS.indexOf(a.el) + 1 : 20;
  let v = cod;
  v = v * 8 + Math.min(grau, 7);
  v = v * 8 + Math.min(hImpl, 7);
  v = v * 8 + (a.carga >= -3 && a.carga <= 3 ? a.carga + 3 : 7);
  v = v * 2 + (a.arom ? 1 : 0);
  return v >>> 0;
}

export function digital(atoms, bonds, nbits = 1024, raio = 2) {
  const n = atoms.length;
  const bits = new Float32Array(nbits);
  if (!n) return bits;
  const viz = atoms.map(() => []);
  for (const b of bonds) { viz[b.a].push([b.b, b.o]); viz[b.b].push([b.a, b.o]); }
  const hs = hidrogenios(atoms, bonds);
  let ident = atoms.map((a, i) => invariante(a, hs[i], viz[i].length));
  for (let i = 0; i < n; i++) bits[ident[i] % nbits] = 1;
  for (let r = 0; r < raio; r++) {
    const novo = new Array(n);
    for (let i = 0; i < n; i++) {
      const vz = viz[i].map(([j, o]) => [o, ident[j]])
        .sort((p, q) => p[0] - q[0] || p[1] - q[1]);
      let h = mix32(2166136261, ident[i]);
      for (const [o, idj] of vz) { h = mix32(h, o); h = mix32(h, idj); }
      novo[i] = h;
    }
    ident = novo;
    for (let i = 0; i < n; i++) bits[ident[i] % nbits] = 1;
  }
  return bits;
}
