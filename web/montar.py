"""Monta a pagina. Junta o codigo, preenche as tabelas com os numeros medidos."""
import json, os, pathlib, re

WEB = pathlib.Path("/home/claude/web")
SITE = WEB / "site"
EXP = pathlib.Path("/home/claude/exp/saida")


def limpa(js):
    js = re.sub(r'^\s*import .*?;\s*$', '', js, flags=re.M)
    js = js.replace("export class", "class").replace("export async function", "async function")
    js = js.replace("export function", "function").replace("export const", "const")
    js = re.sub(r'^\s*export \{[^}]*\};\s*$', '', js, flags=re.M)
    return js


def vir(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def tabela_triagem():
    tri = json.load(open(EXP / "triagem_metricas.json"))
    tox = json.load(open(EXP / "tox_metricas.json"))
    taf = json.load(open(EXP / "tabela_afinidade.json"))
    f = lambda s, m: next(x for x in taf if x["split"] == s and x["modelo"] == m)
    linhas = [
        ("Sintetizabilidade", "ZINC250k, 219 mil de treino",
         f"Spearman {vir(tri['sas']['spearman'], 4)}",
         f"{tri['sas']['n_teste']:,}".replace(",", ".") + " moléculas separadas",
         "Só reproduz o sascorer do RDKit, não mede rota de síntese"),
        ("Alerta toxicológico", "Tox21, 12 ensaios",
         f"AUC médio {vir(tox['media'], 3)}",
         f"{tox['n_teste']:,}".replace(",", ".") + " moléculas, esqueletos disjuntos",
         f"Pior ensaio {vir(min(tox['auc'].values()))}, melhor {vir(max(tox['auc'].values()))}"),
        ("Afinidade", "Davis, 68 moléculas por 442 quinases",
         f"CI {vir(f('molecula nova','interativo')['ci'])} em molécula nova",
         "5.746 pares, molécula nunca vista",
         f"Perde para Tanimoto sem treino, que faz {vir(f('molecula nova','vizinho Tanimoto')['ci'])}"),
    ]
    h = ['<div class="rolagem"><table class="dados"><thead><tr>'
         '<th>Modelo</th><th>Dados</th><th>Medida em teste</th><th>Teste</th>'
         '<th>Limite</th></tr></thead><tbody>']
    for l in linhas:
        h.append("<tr>" + "".join(f"<td>{c}</td>" for c in l) + "</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


def tabela_afinidade():
    t = json.load(open(EXP / "tabela_afinidade.json"))
    splits = ["aleatorio", "molecula nova", "proteina nova", "os dois novos"]
    titulos = ["aleatório", "molécula nova", "proteína nova", "os dois novos"]
    mods = ["media do treino", "vizinho Tanimoto", "so ligante", "aditivo", "interativo"]
    nomes = ["Média do treino", "Vizinho Tanimoto", "Só ligante", "Aditivo", "Interativo"]
    f = lambda s, m: next(x for x in t if x["split"] == s and x["modelo"] == m)
    partes = []
    for campo, titulo in (("ci", "Todos os pares"), ("ci_sem_piso", "Só fora do piso de censura")):
        h = [f'<div class="rolagem"><table class="dados"><thead><tr><th>{titulo}</th>'
             + "".join(f"<th>{x}</th>" for x in titulos) + "</tr></thead><tbody>"]
        for m, nome in zip(mods, nomes):
            melhor = {s: max(mods, key=lambda mm: f(s, mm)[campo]) for s in splits}
            cels = []
            for s in splits:
                v = f(s, m)[campo]
                cls = ' class="venceu"' if melhor[s] == m and v > 0.5 else ""
                cels.append(f"<td{cls}>{vir(v)}</td>")
            h.append(f"<tr><td>{nome}</td>" + "".join(cels) + "</tr>")
        h.append("</tbody></table></div>")
        partes.append("".join(h))
    return "".join(partes)


def tabela_gerador():
    arq = EXP / "protocolo_gerador.json"
    if not arq.exists():
        return ""
    g = json.load(open(arq))
    h = ['<div class="rolagem"><table class="dados"><thead><tr><th>Temperatura</th>'
         '<th>Validade RDKit</th><th>Validade navegador</th><th>Unicidade</th>'
         '<th>Novidade</th><th>Diversidade</th></tr></thead><tbody>']
    for r in g:
        h.append(f"<tr><td>{vir(r['temp'],2)}</td><td>{vir(100*r['validade_rdkit'],1)}%</td>"
                 f"<td>{vir(100*r['validade_navegador'],1)}%</td>"
                 f"<td>{vir(100*r['unicidade'],1)}%</td><td>{vir(100*r['novidade'],1)}%</td>"
                 f"<td>{vir(r['diversidade'],3)}</td></tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


def main():
    js = "\n".join(limpa((WEB / n).read_text())
                   for n in ("quimica.js", "digital.js", "motor.js", "triagem.js"))
    js += "\n" + (SITE / "app.js").read_text()
    assert "export " not in js and not re.search(r'^\s*import ', js, re.M)
    pagina = (SITE / "corpo.html").read_text().replace("/*__JS__*/", js)
    pagina = pagina.replace("<!--TABELA_TRIAGEM-->", tabela_triagem())
    pagina = pagina.replace("<!--TABELA_AFINIDADE-->", tabela_afinidade())
    tg = tabela_gerador()
    if tg:
        pagina = pagina.replace("<!--TABELA_GERADOR-->", tg)
    else:
        # sem a medicao feita, a secao inteira sai, em vez de ficar um titulo oco
        a = pagina.index("<!--INI_GERADOR-->"); b = pagina.index("<!--FIM_GERADOR-->")
        pagina = pagina[:a] + pagina[b + len("<!--FIM_GERADOR-->"):]
    (SITE / "pagina.html").write_text(pagina)

    cab = '''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="MolGPT, um transformer de 819.328 parametros que escreve moleculas em SMILES no navegador, com tres modelos de triagem e as metricas medidas de cada um.">
<style>
:root{color-scheme:dark;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
body{margin:0;font:14px system-ui,sans-serif}
img{max-width:100%}
[hidden]{display:none!important}
</style>
'''
    i = pagina.index('<div class="grao"')
    (WEB / "index.html").write_text(cab + pagina[:i] + "</head>\n<body>\n" + pagina[i:]
                                    + "\n</body>\n</html>\n")
    print(f"pagina.html {len(pagina)/1024:.0f} KB, index.html pronto")


if __name__ == "__main__":
    main()
