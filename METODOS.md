# Métodos

Este arquivo existe para que qualquer número desta página possa ser conferido
ou derrubado. Cada seção diz o dado, a separação, a semente, a métrica e o
comando.

Ambiente: Python 3.13, PyTorch CPU, RDKit, scikit-learn, 2 núcleos. Todas as
sementes estão fixas no código.

---

## 1. Gerador

**Dado.** ZINC250k, 249.455 SMILES, canonizados pelo RDKit.

**Separação.** Por molécula inteira, não por posição no fluxo. 6.000 moléculas
sorteadas com `RandomState(1337)` vão para a validação e são empacotadas num
fluxo separado. As outras 243.455 formam o fluxo de treino. A interseção entre
os dois conjuntos de SMILES é conferida e impressa no log, e é zero.

> Correção em relação à versão anterior. Antes o corte era feito no fluxo já
> empacotado, `dados[:n_val]`, o que deixava janelas atravessando a fronteira
> entre moléculas e pedaços da mesma molécula dos dois lados. A perda de
> validação daquela versão não media molécula nova. Os checkpoints anteriores
> a esta correção estão em `treino/saida` e os novos em `treino/saida_v2`.

**Modelo.** Decoder de 4 camadas, 4 cabeças, dimensão 128, janela de 128
tokens, 819.328 parâmetros, peso da cabeça de saída amarrado à embedding de
token. Tokenização por expressão regular de SMILES. Dois tokens de
condicionamento por molécula, faixa de logP e faixa de QED, cada uma em 5
quantis do próprio treino.

**Protocolo de avaliação.** `exp/protocolo_gerador.py`. Cinco mil moléculas
por temperatura, semente 7, mesmo checkpoint.

| métrica | definição |
|---|---|
| validade | `Chem.MolFromSmiles` devolve molécula |
| unicidade | SMILES canônicos distintos dividido pelas válidas |
| novidade | válidas canônicas que não estão nas 249 mil do treino |
| diversidade | 1 menos a Tanimoto média entre pares, Morgan raio 2, 2048 bits, amostra de mil |

A conferência que roda no navegador é separada e mais permissiva. Ela pega
parêntese aberto, anel sem fechar, valência impossível e aromático fora de
anel, e não pega falha de kekulização. As duas aparecem lado a lado de
propósito.

---

## 2. Impressão digital

Os três modelos de triagem treinam em Python e rodam em JavaScript. Para isso
a feature precisa ser idêntica nos dois lados, então nenhum deles usa RDKit
como entrada.

`quim/quimica.py` é um porte fiel do parser que roda no navegador, e
`web/digital.js` é a mesma impressão circular do lado JavaScript. Hash de 32
bits com a mesma aritmética nas duas linguagens, invariante por átomo com
elemento, grau, hidrogênios implícitos, carga e aromaticidade, duas rodadas de
absorção de vizinhos, dobrada em 1024 ou 2048 bits.

**Conferência.** 400 moléculas sorteadas do ZINC, bits comparados um a um.
Zero divergências. O teste está em `exp/` e roda em segundos.

O RDKit aparece só como rótulo, nunca como entrada.

---

## 3. Sintetizabilidade

**Dado.** ZINC250k, coluna SAS, que é o sascorer do RDKit. Conferido contra o
`sascorer` em 400 moléculas, erro médio 0,0000.

**Separação.** Aleatória por molécula, 219.520 treino e 29.935 teste,
`RandomState(0)`.

**Modelo.** 1024 para 128 para 64 para 1, GELU.

**Resultado.** Spearman 0,9685, erro absoluto médio 0,145, e 97,3 por cento
das moléculas dentro de 0,5 ponto.

**Limite.** Isso reproduz uma regra que já existe. Não mede rota de síntese,
não conhece reagente e não sabe preço.

---

## 4. Alerta toxicológico

**Dado.** Tox21, 7.815 moléculas, 12 ensaios, 82,9 por cento dos rótulos
presentes.

**Separação.** Por esqueleto de Murcko, disjunta. Os esqueletos maiores vão
para o treino e os raros para o teste, que é o caso difícil. 70 por cento
treino, 10 validação, 20 teste.

**Escolha de configuração.** Quatro configurações comparadas na validação. A
escolhida foi 1024 bits, 256 e 64 unidades, dropout 0,3, 80 épocas. O teste
foi medido uma única vez, depois da escolha, com três sementes.

**Resultado.** AUC médio 0,709.

| ensaio | AUC |
|---|---|
| NR-AR-LBD | 0,811 |
| SR-MMP | 0,753 |
| NR-AhR | 0,746 |
| NR-AR | 0,731 |
| NR-ER-LBD | 0,730 |
| NR-PPAR-gamma | 0,722 |
| SR-ARE | 0,683 |
| SR-HSE | 0,678 |
| SR-ATAD5 | 0,679 |
| NR-Aromatase | 0,664 |
| SR-p53 | 0,657 |
| NR-ER | 0,653 |

**Limite.** Doze ensaios in vitro não são toxicidade. Um AUC de 0,65 quer
dizer que em cada cem pares de moléculas, uma ativa e uma inativa, o modelo
ordena certo em sessenta e cinco.

---

## 5. Afinidade

**Dado.** Davis, 68 moléculas por 442 quinases, 30.056 pares. 69,6 por cento
dos pares estão exatamente no piso de pKd 5,0. Isso não é medida, é limite de
ensaio, e por isso toda métrica aparece duas vezes, com e sem esses pares.

O arquivo `Y` do Davis é um pickle baixado. Ele é lido com um unpickler
restrito, que só aceita classes do numpy.

**Separações.** Quatro, três sementes cada.

| nome | regra |
|---|---|
| aleatório | 80 por cento dos pares sorteados, mesma molécula e mesma proteína dos dois lados |
| molécula nova | 20 por cento das moléculas reservadas, nenhum par delas no treino |
| proteína nova | 20 por cento das proteínas reservadas |
| os dois novos | teste só com par de molécula reservada e proteína reservada |

**Modelos.**

| modelo | o que é |
|---|---|
| média do treino | piso, prevê a média |
| vizinho Tanimoto | média ponderada dos 5 pares de treino com a mesma proteína e molécula mais parecida |
| só ligante | rede sem nenhuma informação de proteína |
| aditivo | f(molécula) mais g(proteína), não representa o par |
| interativo | cabeça sobre u, v e o produto u vezes v |

Entrada da molécula, impressão digital própria de 1024 bits. Entrada da
proteína, composição de dipeptídeos, 400 features. Treino com AdamW, 40
épocas, lote 512, cosseno.

**Resultado.** Índice de concordância, média de três sementes.

| modelo | aleatório | molécula nova | proteína nova | os dois novos |
|---|---|---|---|---|
| média do treino | 0,500 | 0,500 | 0,500 | 0,500 |
| vizinho Tanimoto | 0,721 | **0,762** | 0,500 | 0,500 |
| só ligante | 0,750 | 0,605 | 0,752 | 0,599 |
| aditivo | 0,788 | 0,697 | 0,771 | 0,661 |
| interativo | **0,819** | 0,700 | **0,781** | **0,663** |

Só nos pares fora do piso de censura.

| modelo | aleatório | molécula nova | proteína nova | os dois novos |
|---|---|---|---|---|
| vizinho Tanimoto | 0,607 | **0,641** | 0,500 | 0,500 |
| só ligante | 0,587 | 0,545 | 0,588 | 0,556 |
| aditivo | 0,624 | 0,595 | 0,607 | 0,584 |
| interativo | **0,640** | 0,599 | 0,616 | 0,582 |

**Leitura.** A cabeça interativa rompe o teto aditivo no split aleatório e em
proteína nova. Em molécula nova o ganho some, e a busca por semelhança de
Tanimoto, sem treino nenhum, vence todas as redes. Como tudo que o gerador
escreve é molécula inédita, este modelo entra na página como prior fraco, não
como previsor.

**Decomposição de variância.** `exp/teto_proteico.py`. Decomposição de dois
fatores na matriz 68 por 442. Efeito da molécula 22,6 por cento, efeito da
proteína 12,6 por cento, específico do par 64,7 por cento. Tirando o piso, o
termo de par sobe para 73,3 por cento. É isso que explica por que um modelo
que apenas concatena as duas descrições empaca.

**Teto do identificador.** No mesmo script, trocar o descritor de proteína por
um indicador da identidade dela, que é informação perfeita para proteína vista
no treino, mantém o modelo aditivo em 0,790. O limite é a forma de combinar,
não a representação.

---

## 6. Dominio de aplicação

O Davis tem 68 ligantes, quase todos inibidores de quinase. A Tanimoto média
entre eles é 0,145. O gerador escreve molécula tipo ZINC. Para cada molécula
gerada a página calcula a Tanimoto máxima contra essas 68 e marca como fora do
domínio abaixo de 0,3, que é o caso da maioria.

O número de afinidade aparece como percentil contra duas mil moléculas do ZINC
avaliadas no mesmo alvo, e não como medida absoluta, exatamente porque a
previsão pontual nesse regime não se sustenta.

---

## 7. Conferência numérica entre Python e JavaScript

Todo modelo que roda no navegador tem um caso de referência gerado pelo
PyTorch e conferido pelo JavaScript.

| modelo | erro máximo |
|---|---|
| gerador, logits | 3,4e-3, mesmo argmax |
| sintetizabilidade | 1,1e-4 |
| toxicologia, 12 logits | 6,1e-3 |
| afinidade | 1,7e-3 |

Esse é o erro de arredondamento do float16 em que os pesos são exportados.
Isso prova equivalência numérica entre as duas implementações, e só isso.
