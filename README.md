# MolGPT

Um transformer de 819.328 parâmetros que escreve moléculas em SMILES, rodando
inteiro no navegador, mais três modelos de triagem que tentam responder o que
o gerador não sabe, e a medição de cada um deles.

A página está em **[molgpt.vercel.app](https://molgpt.vercel.app)**. Tudo que
aparece lá é calculado no seu navegador. Não existe servidor de inferência.

O que distingue este repositório de uma replicação comum não é o modelo, é a
medição. Os métodos completos, com separações, sementes e comandos, estão em
**[METODOS.md](METODOS.md)**.

---

## O que roda no navegador

Os pesos chegam em float16 codificados em base64. A inferência é token a
token, com cache de chave e valor, cerca de 3 milissegundos por token, então o
navegador entrega um token por quadro e a molécula cresce na tela no ritmo em
que o modelo decide. O SMILES é lido por um parser próprio e desenhado por um
relaxamento de forças, sem biblioteca de química.

Cada modelo exportado tem um caso de referência gerado pelo PyTorch e
conferido pelo JavaScript.

| modelo | erro máximo contra o PyTorch |
|---|---|
| gerador, logits | 3,4e-3, mesmo argmax |
| sintetizabilidade | 1,1e-4 |
| toxicologia, 12 logits | 6,1e-3 |
| afinidade | 1,7e-3 |

Esse é o arredondamento do float16, e prova equivalência numérica entre as
duas implementações, nada além disso.

A impressão digital que alimenta os três modelos de triagem é calculada em
Python no treino e em JavaScript no navegador. As duas foram comparadas bit a
bit em 400 moléculas do ZINC, com zero divergências. Sem isso, modelo treinado
em Python rodando em JavaScript é gerador de números bonitos.

---

## O gerador

Protocolo fixo, cinco mil moléculas por temperatura, semente fixa, mesmo
checkpoint.

| temperatura | validade RDKit | validade do navegador | unicidade | novidade | diversidade |
|---|---|---|---|---|---|
| 0,60 | 92,4% | 98,4% | 100,0% | 99,8% | 0,830 |
| 0,70 | 88,5% | 96,7% | 99,9% | 99,7% | 0,847 |
| 0,80 | 83,4% | 93,8% | 99,9% | 99,8% | 0,859 |
| 0,90 | 77,0% | 87,9% | 99,9% | 99,8% | 0,870 |
| 1,00 | 69,5% | 84,3% | 100,0% | 99,9% | 0,880 |

A conferência que roda no navegador é mais permissiva que o RDKit porque não
enxerga falha de kekulização, que é a maior classe de erro do modelo. As duas
aparecem separadas de propósito, aqui e na página.

---

## Os três modelos de triagem

| modelo | dado | medida em teste | limite |
|---|---|---|---|
| Sintetizabilidade | ZINC250k | Spearman 0,9685 em 29,935 moléculas separadas | Só reproduz o sascorer do RDKit, não conhece rota nem reagente |
| Alerta toxicológico | Tox21, 12 ensaios | AUC médio 0,709 em 1,562 moléculas, esqueletos de Murcko disjuntos | Doze ensaios in vitro não são toxicidade |
| Afinidade | Davis, 68 por 442 | CI 0,700 em molécula nova | Perde para busca por semelhança sem treino nenhum |

---

## Afinidade, a tabela inteira

Índice de concordância, três sementes por célula.

| modelo | aleatório | molécula nova | proteína nova | os dois novos |
|---|---|---|---|---|
| Média do treino | 0,500 | 0,500 | 0,500 | 0,500 |
| Vizinho Tanimoto | 0,721 | **0,762** | 0,500 | 0,500 |
| Só ligante | 0,750 | 0,605 | 0,752 | 0,599 |
| Aditivo | 0,788 | 0,697 | 0,771 | 0,661 |
| Interativo | **0,819** | 0,700 | **0,781** | **0,663** |

Só nos pares que não estão no piso de censura de pKd 5,0, que são 69,6 por
cento do Davis e não são medidas e sim limite de ensaio.

| modelo | aleatório | molécula nova | proteína nova | os dois novos |
|---|---|---|---|---|
| Média do treino | 0,500 | 0,500 | 0,500 | 0,500 |
| Vizinho Tanimoto | 0,607 | **0,641** | 0,500 | 0,500 |
| Só ligante | 0,587 | 0,545 | 0,588 | 0,556 |
| Aditivo | 0,624 | 0,595 | 0,607 | **0,584** |
| Interativo | **0,640** | 0,599 | **0,616** | 0,582 |

Três leituras, duas delas contra a arquitetura que eu mesmo propus.

A cabeça interativa rompe o teto aditivo no split aleatório, 0,819 contra
0,788, e em proteína nova, 0,781 contra 0,771. Em molécula nova o ganho some,
0,700 contra 0,697.

Em molécula nova, a busca por semelhança de Tanimoto, que não tem treino
nenhum, faz 0,762 e vence todas as redes. Como tudo que este gerador
escreve é molécula inédita, é esse o regime que vale, e nele eu ainda não
tenho um modelo de afinidade.

Tirando o piso de censura, tudo cai para a faixa de 0,55 a 0,64. Boa parte do
número alto era o modelo acertando que o par não se liga.

Por isso a página mostra afinidade como percentil contra duas mil moléculas do
ZINC no mesmo alvo, com a Tanimoto máxima contra as 68 moléculas do Davis ao
lado. Quase toda molécula gerada cai fora do domínio de treino, com Tanimoto
entre 0,10 e 0,15, e a página diz isso na cara.

---

## Correções desta versão

**Separação de treino e validação.** A versão anterior cortava o fluxo de
tokens já empacotado num ponto fixo, o que deixava janelas atravessando a
fronteira entre moléculas e pedaços da mesma molécula dos dois lados. A perda
de validação não media molécula nova. Agora 6.000 moléculas inteiras são
sorteadas e empacotadas num fluxo separado, e a interseção entre os dois
conjuntos é conferida e impressa no log.

**A interface antiga saiu.** O arquivo `interface/index.html` animava traços
gravados, escolhendo entre sequências já armazenadas. Ele passou a conviver
com a página da raiz, que roda o modelo de verdade, e isso dava a impressão
errada sobre o que o projeto faz. O histórico do git guarda o arquivo.

---

## O que isso não é

Este modelo escreve moléculas plausíveis. Os três modelos de triagem ordenam
essas moléculas, mal, e servem para descartar o obviamente ruim, não para
apontar o bom.

Nenhum número deste repositório é evidência de que uma molécula se liga a uma
proteína, de que ela é fabricável ou de que ela é segura. Gerar molécula,
prever propriedade e descobrir intervenção biológica são três níveis
diferentes de capacidade. Isto aqui está no primeiro, com medição do segundo.

---

## Mapa do repositório

| caminho | o que é |
|---|---|
| `index.html` | a página, montada por `web/montar.py` |
| `molgpt.b64.txt`, `triagem.b64.txt` | pesos em float16, base64 |
| `web/` | motor de inferência, química, impressão digital e triagem em JavaScript |
| `quim/quimica.py` | porte do parser e da impressão digital para Python |
| `exp/` | os experimentos, um arquivo por pergunta |
| `estudo/` | treinamento do gerador e diagnóstico de falhas |
| `molgpt/` | pacote de química, splits, métricas e baselines |
| `METODOS.md` | separações, sementes, métricas e comandos |
