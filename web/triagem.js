/*
 * Os tres modelos de triagem no navegador.
 *
 *   sintetizabilidade   destilado do sascorer do RDKit
 *   alerta toxicologico 12 ensaios do Tox21
 *   afinidade           Davis, 442 quinases, cabeca interativa
 *
 * Todos comem a MESMA impressao digital que o Python calculou no treino, o
 * que esta conferido bit a bit. Cada numero sai acompanhado do quanto a
 * molecula se parece com o que o modelo viu, porque previsao fora do
 * dominio de treino nao e previsao, e chute com casas decimais.
 */
import { decodeF16 } from "./motor.js";

function geluT(x) {
  // aproximacao por tangente hiperbolica, erro menor que 1e-3 e suficiente aqui
  const c = 0.7978845608028654;
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}

export class Triagem {
  constructor(meta, pesos) {
    this.m = meta;
    this.w = pesos;
  }

  t(nome) {
    const e = this.m.mapa[nome];
    const n = e.shape.reduce((a, b) => a * b, 1);
    return { v: this.w.subarray(e.off, e.off + n), shape: e.shape };
  }

  /** Uma camada linear. W e [saida, entrada], na ordem do PyTorch. */
  linear(x, nome) {
    const W = this.t(nome + ".w"), b = this.t(nome + ".b");
    const [nOut, nIn] = W.shape;
    const out = new Float32Array(nOut);
    for (let i = 0; i < nOut; i++) {
      let s = b.v[i];
      const base = i * nIn;
      for (let j = 0; j < nIn; j++) s += W.v[base + j] * x[j];
      out[i] = s;
    }
    return out;
  }

  mlp(x, prefixo, n) {
    let h = x;
    for (let k = 0; k < n; k++) {
      h = this.linear(h, `${prefixo}.${k}`);
      if (k < n - 1) for (let i = 0; i < h.length; i++) h[i] = geluT(h[i]);
    }
    return h;
  }

  sintetizabilidade(fp1024) {
    return this.mlp(fp1024, "sas", this.m.sas.camadas)[0];
  }

  /** Devolve os 12 logits do Tox21, na ordem de meta.tox.ensaios. */
  toxicologia(fpTox) {
    return this.mlp(fpTox, "tox", this.m.tox.camadas);
  }

  /** Codifica a molecula uma vez, depois cada alvo custa so a cabeca. */
  codifica(fp1024) {
    return this.mlp(fp1024, "af_f", this.m.afinidade.camadas_f);
  }

  afinidade(u, alvo) {
    const V = this.t("af_V");
    const d = V.shape[1];
    const x = new Float32Array(d * 3);
    for (let i = 0; i < d; i++) {
      const v = V.v[alvo * d + i];
      x[i] = u[i]; x[d + i] = v; x[2 * d + i] = u[i] * v;
    }
    return this.mlp(x, "af_h", this.m.afinidade.camadas_h)[0];
  }

  /** Posicao do valor dentro dos quantis de referencia, de 0 a 100. */
  percentil(q, valor) {
    const qs = this.m.quantis;
    if (valor <= q[0]) return 0;
    if (valor >= q[q.length - 1]) return 100;
    for (let i = 1; i < q.length; i++) {
      if (valor <= q[i]) {
        const f = (valor - q[i - 1]) / Math.max(1e-9, q[i] - q[i - 1]);
        return qs[i - 1] + f * (qs[i] - qs[i - 1]);
      }
    }
    return 100;
  }

  quantisAfinidade(alvo) {
    const Q = this.t("q_afinidade");
    const n = Q.shape[1];
    return Q.v.subarray(alvo * n, alvo * n + n);
  }

  quantisTox(ensaio) {
    const Q = this.t("q_tox");
    const n = Q.shape[1];
    return Q.v.subarray(ensaio * n, ensaio * n + n);
  }

  /**
   * Dominio de aplicacao. Tanimoto maxima contra as 68 moleculas do Davis.
   * O gerador escreve molecula tipo ZINC e o Davis e quase todo inibidor de
   * quinase, entao esse numero costuma ser baixo, e e exatamente isso que
   * o usuario precisa saber antes de acreditar na afinidade.
   */
  tanimotoDavis(fp1024) {
    if (!this._davis) {
      const bruto = atob(this.m.davis_bits_b64);
      const bytes = 1024 / 8;
      this._davis = [];
      for (let i = 0; i < this.m.davis_n; i++) {
        const u = new Uint8Array(bytes);
        for (let k = 0; k < bytes; k++) u[k] = bruto.charCodeAt(i * bytes + k);
        this._davis.push(u);
      }
    }
    let melhor = 0;
    let na = 0;
    for (let i = 0; i < fp1024.length; i++) if (fp1024[i]) na++;
    for (const u of this._davis) {
      let comum = 0, nb = 0;
      for (let k = 0; k < u.length; k++) {
        let byte = u[k];
        while (byte) { nb += byte & 1; byte >>= 1; }
      }
      for (let i = 0; i < fp1024.length; i++)
        if (fp1024[i] && (u[i >> 3] >> (i & 7)) & 1) comum++;
      const t = comum / Math.max(1, na + nb - comum);
      if (t > melhor) melhor = t;
    }
    return melhor;
  }
}

export async function carregarTriagem(base = "") {
  const [meta, txt] = await Promise.all([
    fetch(base + "triagem.json").then(r => r.json()),
    fetch(base + "triagem.b64.txt").then(r => r.text()),
  ]);
  const bruto = atob(txt.trim());
  const u8 = new Uint8Array(bruto.length);
  for (let i = 0; i < bruto.length; i++) u8[i] = bruto.charCodeAt(i);
  return new Triagem(meta, decodeF16(u8.buffer));
}
