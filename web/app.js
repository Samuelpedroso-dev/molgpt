/* ====== pagina, estado e desenho ====== */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const PRETO = "#000000", BRANCO = "#ffffff";
  const semMov = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const tela = $("tela"), ctx = tela.getContext("2d");
  let W = 0, H = 0, dpr = 1;

  function medir() {
    const r = tela.getBoundingClientRect();
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height));
    tela.width = Math.round(W * dpr); tela.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  new ResizeObserver(medir).observe(tela);
  medir();

  /* ---- campo de fundo, filamentos lentos ---- */
  const campo = [];
  for (let i = 0; i < 46; i++)
    campo.push({ x: Math.random(), y: Math.random(), vx: (Math.random() - .5) * 6e-5,
                 vy: (Math.random() - .5) * 6e-5 });

  function fundo(dt) {
    ctx.fillStyle = PRETO; ctx.fillRect(0, 0, W, H);
    const g = ctx.createRadialGradient(W * .5, H * .44, 0, W * .5, H * .44, Math.max(W, H) * .62);
    g.addColorStop(0, "rgba(255,255,255,.055)"); g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    if (!semMov) for (const p of campo) {
      p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.x < 0) p.x += 1; if (p.x > 1) p.x -= 1;
      if (p.y < 0) p.y += 1; if (p.y > 1) p.y -= 1;
    }
    ctx.lineWidth = 1;
    for (let i = 0; i < campo.length; i++) {
      const a = campo[i];
      ctx.fillStyle = "rgba(255,255,255,.11)";
      ctx.fillRect(a.x * W, a.y * H, 1.2, 1.2);
      for (let j = i + 1; j < campo.length; j++) {
        const b = campo[j];
        const dx = (a.x - b.x) * W, dy = (a.y - b.y) * H;
        const d = Math.hypot(dx, dy);
        if (d > 118) continue;
        ctx.strokeStyle = `rgba(255,255,255,${(1 - d / 118) * .05})`;
        ctx.beginPath(); ctx.moveTo(a.x * W, a.y * H); ctx.lineTo(b.x * W, b.y * H); ctx.stroke();
      }
    }
  }

  /* ---- desenho da molecula ---- */
  const vista = { esc: 30, cx: 0, cy: 0, pronta: false };

  function enquadra(pos, destino, pad) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of pos) {
      if (p.x < x0) x0 = p.x; if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y; if (p.y > y1) y1 = p.y;
    }
    const lw = Math.max(.9, x1 - x0), lh = Math.max(.9, y1 - y0);
    const esc = Math.max(9, Math.min(92, Math.min((W - pad * 2) / lw, (H - pad * 2) / lh)));
    const alvo = { esc, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
    if (!destino.pronta) { Object.assign(destino, alvo); destino.pronta = true; return destino; }
    const k = semMov ? 1 : .11;
    destino.esc += (alvo.esc - destino.esc) * k;
    destino.cx += (alvo.cx - destino.cx) * k;
    destino.cy += (alvo.cy - destino.cy) * k;
    return destino;
  }

  function rotulo(a, h) {
    if (a.el === "C" && !a.carga) return null;
    let t = a.el;
    if (h > 0 && a.el !== "C") t += "H" + (h > 1 ? h : "");
    if (a.carga > 0) t += a.carga > 1 ? "+" + a.carga : "+";
    if (a.carga < 0) t += a.carga < -1 ? "-" + (-a.carga) : "-";
    return t;
  }

  /* desenha num contexto qualquer, usado na tela grande e nas tiras */
  function pintaMolecula(c, lar, alt, mol, v, agora, novo) {
    const { atoms, bonds, pos, hs } = mol;
    if (!atoms.length) return;
    const px = p => ({ x: lar / 2 + (p.x - v.cx) * v.esc, y: alt / 2 + (p.y - v.cy) * v.esc });
    const viz = atoms.map(() => []);
    for (const b of bonds) { viz[b.a].push(b.b); viz[b.b].push(b.a); }
    const alfa = i => {
      if (!mol.nasc) return 1;
      return Math.max(0, Math.min(1, (agora - (mol.nasc[i] || 0)) / 260));
    };
    const lg = Math.max(1.1, v.esc * .055);

    c.lineCap = "round";
    for (const b of bonds) {
      const al = Math.min(alfa(b.a), alfa(b.b));
      if (al <= .02) continue;
      const A = px(pos[b.a]), B = px(pos[b.b]);
      c.strokeStyle = `rgba(255,255,255,${.93 * al})`;
      c.lineWidth = lg;
      const dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) || 1;
      const ux = dx / d, uy = dy / d, nx = -uy, ny = ux;
      // encurta a linha onde tem rotulo, para a letra nao ficar riscada
      const ra = rotulo(atoms[b.a], hs[b.a]) ? v.esc * .26 : 0;
      const rb = rotulo(atoms[b.b], hs[b.b]) ? v.esc * .26 : 0;
      const a0 = { x: A.x + ux * ra, y: A.y + uy * ra };
      const b0 = { x: B.x - ux * rb, y: B.y - uy * rb };
      const sep = Math.max(2, v.esc * .12);
      const linha = (o, frac) => {
        const m = frac || 1;
        const mx = (a0.x + b0.x) / 2, my = (a0.y + b0.y) / 2;
        c.beginPath();
        c.moveTo(mx + (a0.x - mx) * m + nx * o, my + (a0.y - my) * m + ny * o);
        c.lineTo(mx + (b0.x - mx) * m + nx * o, my + (b0.y - my) * m + ny * o);
        c.stroke();
      };
      if (b.o === 2) { linha(sep * .5); linha(-sep * .5); }
      else if (b.o === 3) { linha(0); linha(sep * .85); linha(-sep * .85); }
      else if (b.o === 4) {
        linha(0);
        const mx = (pos[b.a].x + pos[b.b].x) / 2, my = (pos[b.a].y + pos[b.b].y) / 2;
        // centro local do anel, media dos vizinhos dos dois atomos da ligacao
        let vx = 0, vy = 0, nv = 0;
        for (const k of viz[b.a]) if (k !== b.b) { vx += pos[k].x; vy += pos[k].y; nv++; }
        for (const k of viz[b.b]) if (k !== b.a) { vx += pos[k].x; vy += pos[k].y; nv++; }
        const alvoX = nv ? vx / nv : mx, alvoY = nv ? vy / nv : my;
        const para = Math.sign((alvoX - mx) * nx + (alvoY - my) * ny) || 1;
        c.lineWidth = lg * .8;
        c.strokeStyle = `rgba(255,255,255,${.62 * al})`;
        linha(-para * sep * .95, .64);
      } else linha(0);
    }

    const fs = Math.max(9, Math.min(23, v.esc * .42));
    c.font = `500 ${fs}px "JetBrains Mono", ui-monospace, monospace`;
    c.textAlign = "center"; c.textBaseline = "middle";
    atoms.forEach((a, i) => {
      const t = rotulo(a, hs[i]);
      const al = alfa(i);
      if (al <= .02) return;
      const P = px(pos[i]);
      if (t) {
        const w = c.measureText(t).width;
        c.fillStyle = PRETO;
        c.beginPath();
        c.ellipse(P.x, P.y, w / 2 + fs * .2, fs * .66, 0, 0, 6.2832);
        c.fill();
        c.fillStyle = `rgba(255,255,255,${al})`;
        c.fillText(t, P.x, P.y + fs * .04);
      }
      if (i === novo && !semMov) {
        const f = (agora % 900) / 900;
        c.strokeStyle = `rgba(255,255,255,${(1 - f) * .5 * al})`;
        c.lineWidth = 1;
        c.beginPath(); c.arc(P.x, P.y, fs * .5 + f * fs * 1.5, 0, 6.2832); c.stroke();
      }
    });
  }

  /* ---- montagem do grafo a partir do SMILES ---- */
  function monta(smiles, anterior, iters) {
    const { atoms, bonds } = parseSmiles(smiles);
    const pos = atoms.length ? layout(atoms, bonds, iters, anterior ? anterior.pos : null) : [];
    const hs = hidrogenios(atoms, bonds);
    return { atoms, bonds, pos, hs };
  }

  /* ---- estado ---- */
  const E = {
    modelo: null, tri: null, gen: null, smiles: "", mol: null, nasc: [], novo: -1,
    gerando: false, loop: true, temp: .8, faixa: 2, alvo: 0,
    total: 0, aceitas: 0, msTok: [], proxTok: 0, fim: 0, ref: true,
  };

  /* ---- triagem, os tres modelos ---- */
  const pct = x => Math.max(0, Math.min(100, x));

  function limpaTriagem() {
    for (const id of ["t-sas", "t-tox", "t-af"]) $(id).textContent = "\u2014";
    for (const id of ["t-sasb", "t-toxb", "t-afb"]) $(id).style.width = "0%";
    $("t-dom").textContent = "\u2014";
    $("t-dom").className = "dom";
    $("t-afp").textContent = "";
    $("t-toxn").textContent = "";
  }

  function rodaTriagem(mol) {
    const T = E.tri;
    if (!T || !mol || mol.atoms.length < 2) return;
    const fp = digital(mol.atoms, mol.bonds, 1024);
    const fpt = T.m.tox.nin === 1024 ? fp : digital(mol.atoms, mol.bonds, T.m.tox.nin);

    // sintetizabilidade, escala de 1 a 10
    const sas = T.sintetizabilidade(fp);
    $("t-sas").textContent = sas.toFixed(2).replace(".", ",");
    $("t-sasb").style.width = pct((sas - 1) / 9 * 100) + "%";
    const med = T.m.sas.zinc.mediana;
    $("t-sasz").style.left = pct((med - 1) / 9 * 100) + "%";
    $("t-sasn").textContent = "traço, mediana do ZINC " + med.toFixed(2).replace(".", ",");

    // toxicologia, o ensaio de percentil mais alto
    const lg = T.toxicologia(fpt);
    let melhor = 0, mp = -1;
    for (let j = 0; j < lg.length; j++) {
      const q = T.percentil(T.quantisTox(j), lg[j]);
      if (q > mp) { mp = q; melhor = j; }
    }
    const ens = T.m.tox.ensaios[melhor];
    $("t-toxe").textContent = "Alerta toxicológico, " + ens;
    $("t-tox").textContent = mp.toFixed(0) + "\u00ba";
    $("t-toxb").style.width = pct(mp) + "%";
    const auc = (T.m.tox.auc_3_sementes || {})[ens];
    $("t-toxn").textContent = auc ? "AUC deste ensaio " + auc.toFixed(3).replace(".", ",") : "";

    // afinidade, com dominio de aplicacao
    E.u = T.codifica(fp);
    E.fp = fp;
    pintaAfinidade();
  }

  function pintaAfinidade() {
    const T = E.tri;
    if (!T || !E.u) return;
    const pk = T.afinidade(E.u, E.alvo);
    const q = T.percentil(T.quantisAfinidade(E.alvo), pk);
    $("t-af").textContent = q.toFixed(0) + "\u00ba";
    $("t-afb").style.width = pct(q) + "%";
    $("t-afp").textContent = "pKd " + pk.toFixed(2).replace(".", ",");
    const tan = T.tanimotoDavis(E.fp);
    const fora = tan < 0.3;
    $("t-dom").textContent = "Tanimoto " + tan.toFixed(2).replace(".", ",") +
      (fora ? ", fora do domínio" : ", dentro");
    $("t-dom").className = "dom" + (fora ? " fora" : "");
  }

  function propsNaTela(mol, parcial) {
    if (!mol || !mol.atoms.length) return;
    const p = propriedades(mol.atoms, mol.bonds);
    $("p-form").textContent = p.formula || "—";
    $("p-mw").textContent = p.mw.toFixed(2) + (parcial ? "" : "");
    $("p-at").textContent = p.pesados;
    $("p-an").textContent = p.aneis;
    $("p-hbd").textContent = p.hbd;
    $("p-hba").textContent = p.hba;
    $("p-ar").textContent = p.aromaticos;
  }

  function veredito(texto, ok) {
    const v = $("vered");
    v.className = "vered" + (ok === true ? " ok" : ok === false ? " ruim" : "");
    $("v-motivo").textContent = texto;
  }

  function medidas() {
    $("m-ger").textContent = E.total;
    $("m-taxa").textContent = E.total ? (100 * E.aceitas / E.total).toFixed(1).replace(".", ",") + "%" : "—";
    if (E.msTok.length) {
      const m = E.msTok.reduce((a, b) => a + b, 0) / E.msTok.length;
      $("m-tok").textContent = m.toFixed(1).replace(".", ",");
    }
  }

  /* ---- tiras da galeria ---- */
  const guardadas = [];
  function guarda(mol, smiles) {
    guardadas.unshift({ mol, smiles });
    if (guardadas.length > 12) guardadas.pop();
    const alvo = $("tiras");
    alvo.textContent = "";
    for (const g of guardadas) {
      const b = document.createElement("button");
      b.className = "tira"; b.title = g.smiles;
      const cv = document.createElement("canvas");
      const lado = 118, d = Math.min(2, window.devicePixelRatio || 1);
      cv.width = lado * d; cv.height = lado * d;
      const c2 = cv.getContext("2d"); c2.setTransform(d, 0, 0, d, 0, 0);
      c2.fillStyle = PRETO; c2.fillRect(0, 0, lado, lado);
      const v = { esc: 30, cx: 0, cy: 0, pronta: false };
      const antes = [W, H];
      W = lado; H = lado; enquadra(g.mol.pos, v, 22); W = antes[0]; H = antes[1];
      pintaMolecula(c2, lado, lado, { ...g.mol, nasc: null }, v, 0, -1);
      const i = document.createElement("i"); i.textContent = g.smiles;
      b.appendChild(cv); b.appendChild(i);
      b.addEventListener("click", () => inspeciona(g));
      alvo.appendChild(b);
    }
  }

  function inspeciona(g) {
    E.gerando = false; E.loop = false; $("loop").setAttribute("aria-pressed", "false");
    E.ref = false; E.gen = null;
    E.smiles = g.smiles;
    E.mol = { ...g.mol, nasc: null };
    E.novo = -1; vista.pronta = false;
    $("saida").textContent = g.smiles;
    $("cursor").style.display = "none";
    $("estado").textContent = "Inspeção";
    document.querySelector(".fita .rot").textContent = "SMILES, guardada nesta sessão";
    propsNaTela(E.mol);
    const v = valida(g.smiles);
    veredito(v.ok ? "Aceita na conferência" : "Rejeitada, " + v.motivo, v.ok);
    rodaTriagem(E.mol);
  }

  /* ---- ciclo de geracao ---- */
  function comeca() {
    if (!E.modelo) return;
    E.gen = E.modelo.fluxo({ temp: E.temp, faixaLogP: E.faixa, maxTokens: 110 });
    E.smiles = ""; E.mol = null; E.nasc = []; E.novo = -1; E.gerando = true; E.ref = false;
    vista.pronta = false;
    $("saida").textContent = "";
    $("cursor").style.display = "";
    $("estado").textContent = "Gerando ao vivo";
    limpaTriagem();
    document.querySelector(".fita .rot").textContent = "SMILES, saída bruta do modelo";
    veredito("Escrevendo", null);
  }

  function passoDeGeracao(agora) {
    const t0 = performance.now();
    const r = E.gen.next();
    const dt = performance.now() - t0;
    E.msTok.push(dt); if (E.msTok.length > 240) E.msTok.shift();

    if (r.done) {
      E.gerando = false; E.gen = null;
      $("cursor").style.display = "none";
      E.mol = monta(E.smiles, E.mol, 320);
      E.mol.nasc = E.nasc;
      vista.pronta = false;
      const v = valida(E.smiles);
      E.total++; if (v.ok) E.aceitas++;
      veredito(v.ok ? "Aceita na conferência" : "Rejeitada, " + v.motivo, v.ok);
      $("estado").textContent = v.ok ? "Aceita" : "Rejeitada";
      propsNaTela(E.mol);
      if (v.ok) rodaTriagem(E.mol); else limpaTriagem();
      medidas();
      if (v.ok && E.mol.atoms.length > 2) guarda(E.mol, E.smiles);
      E.fim = agora;
      return;
    }
    const { sim, p } = r.value;
    E.smiles += sim;
    const esp = document.createElement("span");
    esp.textContent = sim;
    esp.style.opacity = (.45 + .55 * Math.min(1, p * 1.4)).toFixed(2);
    $("saida").appendChild(esp);
    const antes = E.mol ? E.mol.atoms.length : 0;
    E.mol = monta(E.smiles, E.mol, 34);
    E.mol.nasc = E.nasc;
    for (let i = antes; i < E.mol.atoms.length; i++) E.nasc[i] = agora;
    E.novo = E.mol.atoms.length - 1;
    propsNaTela(E.mol, true);
  }

  let ultimo = performance.now();
  function quadro(agora) {
    const dt = Math.min(64, agora - ultimo); ultimo = agora;
    fundo(dt);
    if (E.gerando && E.gen && agora >= E.proxTok) {
      E.proxTok = agora + 30;
      passoDeGeracao(agora);
    }
    if (E.mol && E.mol.atoms.length) {
      enquadra(E.mol.pos, vista, Math.max(34, Math.min(W, H) * .11));
      pintaMolecula(ctx, W, H, E.mol, vista, agora, E.gerando ? E.novo : -1);
    }
    if (!E.gerando && E.loop && E.modelo && E.fim && agora - E.fim > 1500) comeca();
    requestAnimationFrame(quadro);
  }

  /* ---- controles ---- */
  const FAIXAS = [0, 1, 2, 3, 4];
  let logps = { 0: .8, 1: 1.97, 2: 2.55, 3: 3.12, 4: 3.73 };
  const cx = $("faixas");
  FAIXAS.forEach(f => {
    const b = document.createElement("button");
    b.textContent = f; b.setAttribute("aria-pressed", String(f === E.faixa));
    b.setAttribute("aria-label", "faixa " + f);
    b.addEventListener("click", () => {
      E.faixa = f;
      [...cx.children].forEach((c, i) => c.setAttribute("aria-pressed", String(i === f)));
      $("v-faixa").textContent = f;
      $("v-logp").textContent = (logps[f] ?? 0).toFixed(2).replace(".", ",");
    });
    cx.appendChild(b);
  });
  $("temp").addEventListener("input", e => {
    E.temp = parseFloat(e.target.value);
    $("v-temp").textContent = E.temp.toFixed(2).replace(".", ",");
    const k = E.temp.toFixed(1);
    const vm = E.modelo && E.modelo.m.validade_medida && E.modelo.m.validade_medida[k];
    if (vm) $("m-rdkit").textContent = (100 * vm).toFixed(1).replace(".", ",") + "%";
  });
  $("gerar").addEventListener("click", () => { if (E.modelo) { E.fim = 0; comeca(); } });
  $("loop").addEventListener("click", () => {
    E.loop = !E.loop;
    $("loop").setAttribute("aria-pressed", String(E.loop));
    if (E.loop && !E.gerando) { E.fim = 0; comeca(); }
  });

  /* ---- referencia enquanto os pesos chegam ---- */
  const ASPIRINA = "CC(=O)Oc1ccccc1C(=O)O";
  E.mol = monta(ASPIRINA, null, 420);
  E.smiles = ASPIRINA;
  $("saida").textContent = ASPIRINA;
  $("cursor").style.display = "none";
  document.querySelector(".fita .rot").textContent = "SMILES, referência, aspirina";
  propsNaTela(E.mol);
  veredito("Referência desenhada pelo layout daqui", null);
  requestAnimationFrame(quadro);

  /* ---- carregamento dos pesos ---- */
  /*
   * Os pesos vao em base64 num arquivo de texto porque e o formato que tanto
   * o GitHub Pages quanto o hospedeiro do artifact servem sem combinacao
   * previa. Sao 1,6 MB de float16 virando 2,2 MB de texto, decodificados
   * uma vez no carregamento.
   */
  async function textoComProgresso(resp) {
    const tam = parseInt(resp.headers.get("content-length") || "0", 10);
    if (!resp.body || !tam) return resp.text();
    const leitor = resp.body.getReader();
    const dec = new TextDecoder();
    let texto = "", lidos = 0;
    for (;;) {
      const { done, value } = await leitor.read();
      if (done) break;
      lidos += value.length;
      texto += dec.decode(value, { stream: true });
      $("estado").textContent = "Carregando pesos " + Math.round(100 * lidos / tam) + "%";
    }
    return texto + dec.decode();
  }

  function deBase64(b64) {
    const bruto = atob(b64.trim());
    const u8 = new Uint8Array(bruto.length);
    for (let i = 0; i < bruto.length; i++) u8[i] = bruto.charCodeAt(i);
    return u8.buffer;
  }

  (async () => {
    try {
      const [meta, resp] = await Promise.all([
        fetch("molgpt.json").then(r => { if (!r.ok) throw new Error("json " + r.status); return r.json(); }),
        fetch("molgpt.b64.txt").then(r => { if (!r.ok) throw new Error("pesos " + r.status); return r; }),
      ]);
      const buf = deBase64(await textoComProgresso(resp));
      if (buf.byteLength !== meta.n_params * 2)
        throw new Error("tamanho inesperado, " + buf.byteLength);
      E.modelo = new MolGPT(meta, decodeF16(buf));
      if (meta.faixas_logp) {
        logps = meta.faixas_logp;
        $("v-logp").textContent = (logps[E.faixa] ?? 0).toFixed(2).replace(".", ",");
      }
      if (meta.passo_treino) $("f-passo").textContent = meta.passo_treino;
      const vm = meta.validade_medida && meta.validade_medida[E.temp.toFixed(1)];
      if (vm) $("m-rdkit").textContent = (100 * vm).toFixed(1).replace(".", ",") + "%";
      comeca();

      // os tres modelos de triagem chegam depois, a bancada ja esta rodando
      try {
        const [tm, ttxt] = await Promise.all([
          fetch("triagem.json").then(r => { if (!r.ok) throw new Error("triagem " + r.status); return r.json(); }),
          fetch("triagem.b64.txt").then(r => { if (!r.ok) throw new Error("pesos da triagem " + r.status); return r.text(); }),
        ]);
        E.tri = new Triagem(tm, decodeF16(deBase64(ttxt)));
        const sel = $("alvo");
        tm.afinidade.proteinas.forEach((nome, i) => {
          const o = document.createElement("option");
          o.value = i; o.textContent = nome;
          sel.appendChild(o);
        });
        const abl = tm.afinidade.proteinas.findIndex(n => n.startsWith("ABL1"));
        E.alvo = abl >= 0 ? abl : 0;
        sel.value = String(E.alvo);
        sel.addEventListener("change", e => { E.alvo = +e.target.value; pintaAfinidade(); });
        if (E.mol) rodaTriagem(E.mol);
      } catch (e2) {
        $("t-dom").textContent = "triagem fora, " + e2.message;
      }
    } catch (err) {
      $("estado").textContent = "Pesos não carregaram";
      $("ponto").style.animation = "none";
      $("ponto").style.opacity = ".3";
      veredito("Os pesos não chegaram, " + err.message, false);
    }
  })();
})();
