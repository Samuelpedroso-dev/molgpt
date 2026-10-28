# web

O que faz a página rodar o modelo no navegador, sem biblioteca nenhuma.

| arquivo | o que é |
|---|---|
| `motor.js` | inferência do transformer, cache de chave e valor, decodificador de float16 |
| `quimica.js` | parser de SMILES, layout 2D por relaxamento, propriedades, conferência de validade |
| `digital.js` | impressão digital circular, a mesma do `quim/quimica.py`, bit a bit |
| `triagem.js` | os três modelos de triagem, percentis e domínio de aplicação |
| `app.js` | a bancada, desenho em canvas e o laço de geração |
| `corpo.html` | marcação e estilo |
| `montar.py` | junta tudo e preenche as tabelas com os números medidos |
| `exportar.py` | exporta o checkpoint do gerador para float16 e base64 |

Para montar a página, `python3 web/montar.py`, que escreve o `index.html` da
raiz. Os pesos ficam na raiz, em `molgpt.b64.txt` e `triagem.b64.txt`, porque
é o formato que o GitHub Pages e o hospedeiro da página servem sem combinação
prévia.
