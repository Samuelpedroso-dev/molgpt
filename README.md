# MolGPT

Um GPT pequeno que aprende a escrever moléculas e uma interface para explorar o que ele inventa. O modelo lê e escreve moléculas no formato SMILES, um pedaço por vez, e aceita pedidos de faixa de logP e de QED antes de gerar. A interface mostra as moléculas geradas como uma rede por similaridade, com estrutura 2D animada, modelo 3D girando e a escrita do GPT com a confiança de cada escolha.

Este projeto faz parte de um estudo maior sobre usar IA para propor candidatas a medicamentos. O que existe aqui é a etapa de geração. Ele não sabe nada sobre alvos nem sobre afinidade, então o que sai são moléculas parecidas com fármacos e não medicamentos.

## O que tem no repositório

O arquivo molgpt.py contém o código completo de dados, modelo, treino, geração e avaliação, e molgpt.ipynb é o mesmo código em notebook para rodar no Colab com GPU. A pasta modelo guarda o checkpoint do modelo pequeno treinado, a pasta resultados guarda as métricas e o registro do treino, e a pasta interface tem uma única página HTML que funciona offline e pode ser aberta direto no navegador.

## Resultados do modelo pequeno

O modelo tem 0,8 milhão de parâmetros, 4 camadas e foi treinado por 2600 rodadas no ZINC250k, em cerca de 28 minutos numa CPU de 2 núcleos. A perda final ficou em 0,77 no treino e na validação, o que indica que ele aprendeu sem decorar.

Foram geradas 4000 moléculas e comparadas com uma amostra real do ZINC250k, medidas com o RDKit. A validade ficou em 56 por cento, e é a parte fraca esperada para um modelo pequeno com treino curto. Entre as válidas, todas eram únicas e quase todas novas, fora do conjunto de treino. A diversidade foi de 0,88 contra 0,88 do ZINC real, o QED médio de 0,73 contra 0,74, o logP médio de 2,43 contra 2,38, 97 por cento passam na regra de Lipinski contra 99 por cento e 2 por cento têm alertas PAINS contra 3 por cento.

O controle por faixa de logP funcionou. Pedindo da faixa mais baixa para a mais alta, o logP médio das moléculas devolvidas foi de 0,75 até 4,17, subindo a cada faixa. Os números completos estão em resultados/metricas.json.

## Como rodar

Instale as dependências com `pip install rdkit torch pandas numpy matplotlib`. Para treinar o modelo pequeno em CPU rode `python molgpt.py cpu`, que baixa o ZINC250k sozinho e leva cerca de meia hora. Para treinar a versão maior, de 5 milhões de parâmetros, rode `python molgpt.py` numa máquina com GPU ou abra molgpt.ipynb no Colab com a GPU T4 ligada, o que deve levar de 15 a 25 minutos. Essa estimativa para GPU não foi medida.

## Como funciona

Cada molécula vira uma sequência de pedaços, chamados tokens, que começa com um marcador de início e dois marcadores de controle com a faixa de logP e de QED. O GPT aprende a prever o próximo pedaço, e na geração basta dar os marcadores de controle que ele escreve o resto. O código separa as moléculas do ZINC250k em treino e validação e usa o RDKit para checar se o que o modelo escreve é uma molécula válida.

## Limites

O modelo foi treinado só em moléculas do ZINC250k e não conhece proteínas, alvos nem toxicidade. A página de interface mostra moléculas e escritas gravadas desse modelo pequeno e não roda o modelo no navegador. A etapa seguinte é ligar um modelo de afinidade, treinado em Davis ou KIBA, como juiz das moléculas geradas, depois ajustar o GPT por reforço e por fim encaixar as melhores candidatas no bolso de um alvo real.
