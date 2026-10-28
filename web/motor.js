/*
 * Motor de inferencia do MolGPT em JavaScript puro, sem biblioteca nenhuma.
 *
 * O modelo tem 819.328 parametros. Com cache de chave e valor, cada token
 * novo custa uma passada de vetor por matriz, nao a sequencia inteira. Isso
 * e o que permite gerar no navegador em tempo real em vez de mostrar
 * molecula gravada.
 *
 * Pesos em float16, decodificados uma vez no carregamento.
 */

export function decodeF16(buf) {
  const u16 = new Uint16Array(buf);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sgn = (h & 0x8000) ? -1 : 1;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;
    if (exp === 0) out[i] = sgn * 6.103515625e-5 * (frac / 1024);
    else if (exp === 0x1f) out[i] = frac ? NaN : sgn * Infinity;
    else out[i] = sgn * Math.pow(2, exp - 15) * (1 + frac / 1024);
  }
  return out;
}

function erf(x) {
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const gelu = x => 0.5 * x * (1 + erf(x * 0.7071067811865476));

export class MolGPT {
  constructor(meta, pesos) {
    this.m = meta;
    this.w = pesos;
    this.d = meta.n_embd;
    this.H = meta.n_head;
    this.L = meta.n_layer;
    this.hd = this.d / this.H;
    this.V = meta.vocab.length;
    this.stoi = {};
    meta.vocab.forEach((t, i) => { this.stoi[t] = i; });
  }

  t(nome) {
    const e = this.m.mapa[nome];
    const n = e.shape.reduce((a, b) => a * b, 1);
    return this.w.subarray(e.off, e.off + n);
  }

  // out[i] = soma_j W[i*in+j]*x[j] + b[i]
  linear(x, W, b, nIn, nOut, out) {
    for (let i = 0; i < nOut; i++) {
      let s = b ? b[i] : 0;
      const base = i * nIn;
      for (let j = 0; j < nIn; j++) s += W[base + j] * x[j];
      out[i] = s;
    }
    return out;
  }

  layernorm(x, g, b, n, out) {
    let mu = 0;
    for (let i = 0; i < n; i++) mu += x[i];
    mu /= n;
    let va = 0;
    for (let i = 0; i < n; i++) { const dx = x[i] - mu; va += dx * dx; }
    const inv = 1 / Math.sqrt(va / n + 1e-5);
    for (let i = 0; i < n; i++) out[i] = (x[i] - mu) * inv * g[i] + b[i];
    return out;
  }

  novoCache() {
    const c = [];
    for (let l = 0; l < this.L; l++) {
      c.push({
        k: new Float32Array(this.m.block_size * this.d),
        v: new Float32Array(this.m.block_size * this.d),
        n: 0,
      });
    }
    return c;
  }

  /** Um passo. Recebe o id do token e a posicao, devolve os logits. */
  passo(tokId, pos, cache) {
    const d = this.d, H = this.H, hd = this.hd;
    const tok = this.t("tok.weight"), posw = this.t("pos.weight");
    let x = new Float32Array(d);
    for (let i = 0; i < d; i++) x[i] = tok[tokId * d + i] + posw[pos * d + i];

    const h = new Float32Array(d);
    const qkv = new Float32Array(3 * d);
    const att = new Float32Array(d);
    const proj = new Float32Array(d);
    const m1 = new Float32Array(4 * d);
    const m2 = new Float32Array(d);

    for (let l = 0; l < this.L; l++) {
      const p = `blocos.${l}.`;
      this.layernorm(x, this.t(p + "ln1.weight"), this.t(p + "ln1.bias"), d, h);
      this.linear(h, this.t(p + "qkv.weight"), this.t(p + "qkv.bias"), d, 3 * d, qkv);

      const C = cache[l], t = C.n;
      for (let i = 0; i < d; i++) {
        C.k[t * d + i] = qkv[d + i];
        C.v[t * d + i] = qkv[2 * d + i];
      }
      C.n = t + 1;

      const esc = 1 / Math.sqrt(hd);
      for (let hh = 0; hh < H; hh++) {
        const o = hh * hd;
        let mx = -Infinity;
        const sc = new Float32Array(C.n);
        for (let s = 0; s <= t; s++) {
          let dot = 0;
          for (let i = 0; i < hd; i++) dot += qkv[o + i] * C.k[s * d + o + i];
          sc[s] = dot * esc;
          if (sc[s] > mx) mx = sc[s];
        }
        let soma = 0;
        for (let s = 0; s <= t; s++) { sc[s] = Math.exp(sc[s] - mx); soma += sc[s]; }
        for (let i = 0; i < hd; i++) {
          let acc = 0;
          for (let s = 0; s <= t; s++) acc += sc[s] * C.v[s * d + o + i];
          att[o + i] = acc / soma;
        }
      }

      this.linear(att, this.t(p + "proj.weight"), this.t(p + "proj.bias"), d, d, proj);
      for (let i = 0; i < d; i++) x[i] += proj[i];

      this.layernorm(x, this.t(p + "ln2.weight"), this.t(p + "ln2.bias"), d, h);
      this.linear(h, this.t(p + "mlp.0.weight"), this.t(p + "mlp.0.bias"), d, 4 * d, m1);
      for (let i = 0; i < 4 * d; i++) m1[i] = gelu(m1[i]);
      this.linear(m1, this.t(p + "mlp.2.weight"), this.t(p + "mlp.2.bias"), 4 * d, d, m2);
      for (let i = 0; i < d; i++) x[i] += m2[i];
    }

    this.layernorm(x, this.t("ln_f.weight"), this.t("ln_f.bias"), d, h);
    // cabeca compartilha peso com a embedding de token
    const logits = new Float32Array(this.V);
    for (let v = 0; v < this.V; v++) {
      let s = 0;
      const base = v * d;
      for (let i = 0; i < d; i++) s += tok[base + i] * h[i];
      logits[v] = s;
    }
    return logits;
  }

  /** Amostra um token dos logits, com temperatura. Devolve o id e a probabilidade. */
  amostrar(logits, temp) {
    const lg = Float32Array.from(logits);
    lg[this.m.pad] = -1e9;
    lg[this.m.bos] = -1e9;
    let mx = -Infinity;
    for (let i = 0; i < lg.length; i++) { lg[i] /= temp; if (lg[i] > mx) mx = lg[i]; }
    let soma = 0;
    const pr = new Float32Array(lg.length);
    for (let i = 0; i < lg.length; i++) { pr[i] = Math.exp(lg[i] - mx); soma += pr[i]; }
    for (let i = 0; i < pr.length; i++) pr[i] /= soma;
    let r = Math.random(), id = pr.length - 1;
    for (let i = 0; i < pr.length; i++) { r -= pr[i]; if (r <= 0) { id = i; break; } }
    return { id, p: pr[id] };
  }

  /**
   * Gerador token a token. Cada `next()` custa UMA passada de vetor por
   * matriz, cerca de 1,3 ms, entao o navegador pode entregar um token por
   * quadro e a molecula cresce na tela enquanto o modelo decide.
   * `faixaLogP` de 0 a 4 condiciona a lipofilicidade.
   */
  *fluxo({ temp = 0.8, faixaLogP = null, faixaQED = null, maxTokens = 110 } = {}) {
    const cache = this.novoCache();
    const prefixo = [this.m.bos];
    if (faixaLogP !== null && faixaLogP !== undefined) {
      prefixo.push(this.stoi[`<lp${faixaLogP}>`]);
      if (faixaQED !== null && faixaQED !== undefined) prefixo.push(this.stoi[`<q${faixaQED}>`]);
    }
    let logits = null, pos = 0;
    for (const id of prefixo) logits = this.passo(id, pos++, cache);

    for (let n = 0; n < maxTokens && pos < this.m.block_size; n++) {
      const { id, p } = this.amostrar(logits, temp);
      if (id === this.m.eos) return { parado: 'eos' };
      const sim = this.m.vocab[id];
      if (!sim.startsWith('<')) yield { sim, p, n: n + 1 };
      logits = this.passo(id, pos++, cache);
    }
    return { parado: 'limite' };
  }

  /** Gera de uma vez, usando o mesmo fluxo. */
  gerar(opts = {}) {
    const toks = [], confs = [];
    const g = this.fluxo(opts);
    for (const { sim, p } of g) {
      toks.push(sim); confs.push(p);
      if (opts.aoPasso) opts.aoPasso(sim, p, toks.length);
    }
    return { tokens: toks, confiancas: confs, smiles: toks.join('') };
  }
}

export async function carregar(baseUrl = "") {
  const [meta, bin] = await Promise.all([
    fetch(baseUrl + "molgpt.json").then(r => r.json()),
    fetch(baseUrl + "molgpt.bin").then(r => r.arrayBuffer()),
  ]);
  return new MolGPT(meta, decodeF16(bin));
}
