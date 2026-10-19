# MolGPT, estudo de validade em geração de moléculas

Um GPT pequeno que escreve moléculas em SMILES, mais o estudo de por que a validade dele estagna e o que de fato a conserta.

Versão 0.2. O que distingue este repositório de uma replicação comum não é o modelo, é a medição.

## O resultado principal

Começamos com um modelo de 819 mil parâmetros produzindo **56 por cento de moléculas válidas** e com a hipótese de que era subtreino. A hipótese estava errada, e o experimento mostrou isso.

### A validade estagna, a perda não

Mesma arquitetura, treinada do zero com sequências empacotadas em 249.455 moléculas do ZINC250k.

| passo | perda treino | perda validação | validade |
|---|---|---|---|
| 0 | 4.295 | 4.296 | 0.8% |
| 500 | 1.040 | 1.049 | 8.4% |
| 1000 | 0.897 | 0.900 | 28.8% |
| 1500 | 0.842 | 0.840 | 56.8% |
| 2000 | 0.811 | 0.813 | 57.6% |
| 2500 | 0.787 | 0.797 | 58.4% |
| 3000 | 0.777 | 0.779 | 56.0% |

A validade sobe até o passo 1500 e para. A perda continua caindo. Treino e validação permanecem iguais, ou seja, não há sobreajuste, e mesmo assim o modelo não fica mais válido.

### O número 56 era uma escolha de amostragem, não uma propriedade do modelo

O mesmo checkpoint do passo 3000, variando apenas a temperatura.

| temperatura | validade |
|---|---|
| 1.0 | 52.0% |
| 0.9 | 63.2% |
| 0.8 | 68.5% |
| 0.7 | 75.2% |
| 0.6 | 81.2% |

Sem treinar um passo a mais, a validade vai de 52 para 81 por cento. Qualquer validade reportada sem a temperatura ao lado é incomparável.

Amostragem por top k a temperatura 1.0 **não ajuda**, fica entre 52 e 56 por cento. As falhas não vêm da cauda de tokens raros, vêm do corpo da distribuição.

### O que está quebrado é química, não sintaxe

Classificação das moléculas inválidas por tipo de falha, estável em todas as temperaturas testadas.

| falha | fração das inválidas |
|---|---|
| kekulização aromática | 53 a 69% |
| anel não fechado | 15 a 34% |
| parênteses desbalanceados | 12 a 21% |
| valência | 2 a 4% |

Kekulização é a classe dominante com folga. Significa que o modelo escreve um anel com átomos aromáticos minúsculos numa configuração em que não existe alternância de ligações duplas possível.

### A correção que decorre do diagnóstico

Treinar com SMILES **kekulizado**, com ligações duplas explícitas em vez de átomo aromático minúsculo. A classe de falha dominante deixa de ser possível, porque o alfabeto não tem mais aromático.

O efeito na representação é mensurável. O vocabulário cai de 75 para 64 tokens e o fluxo cresce 12 por cento, de 10.554.965 para 11.800.265 tokens.

Esse experimento está em andamento. Os números entram aqui quando terminar, incluindo o caso em que a correção não funcionar.

## Modelo

817.920 parâmetros na versão kekulizada, 819.328 na aromática. A diferença é só o vocabulário, porque a camada de saída compartilha peso com a embedding de token.

Quatro camadas, quatro cabeças, embedding de 128, contexto de 128. Condicionamento por faixa de logP e de QED, cinco faixas em quantis cada.

A engenharia que tornou viável treinar em 2 CPUs foi empacotar as sequências em vez de preencher com token de preenchimento. A versão anterior calculava atenção sobre 80 posições quando a molécula mediana tem perto de 44 tokens, desperdiçando quase metade do cálculo.

## Linhas de base de afinidade em Davis

Trabalho relacionado, no mesmo repositório, porque é a peça que falta para o gerador virar algo útil.

Sessenta combinações de split, modelo e semente. Três achados.

Dos 30.056 pares do Davis, **20.931 estão exatamente em pKd 5.0**, que é o piso de censura e não é medida. São 69.6 por cento do conjunto.

O modelo que **ignora a proteína** alcança índice de concordância 0.752. O modelo completo alcança 0.767. A proteína contribui 0.015.

A degradação em cold start é assimétrica. Proteína nova custa 0.036 de índice de concordância, molécula nova custa 0.136, e ambos novos custam 0.221. No cenário de ambos novos, o melhor modelo melhora 4.7 por cento sobre prever sempre a média, contra 56 por cento no split aleatório.

## Como reproduzir

```bash
pip install torch rdkit scikit-learn pandas numpy

# gerador, versao aromatica de controle
python treinar.py

# gerador, versao kekulizada
KEKULE=1 python treinar.py

# diagnostico de temperatura e de tipo de falha
python diagnostico.py

# linhas de base de afinidade
python davis_baselines.py
```

Semente 42. Em 2 CPUs o gerador leva cerca de 0.80 segundo por passo e as linhas de base de Davis levam 453 segundos.

O pacote `molgpt` traz curadoria de bioatividade, os três splits com portões de vazamento que levantam exceção, métricas de regressão e de triagem, pareamento de decoys e triagem de alvo.

## Limites

O modelo não sabe nada sobre alvo proteico, afinidade ou toxicidade. O que ele produz são pontos de partida, não fármacos.

O estudo de validade usa um modelo pequeno. Nada aqui afirma estado da arte, e as linhas de base publicadas ficam perto de 97 por cento de validade.

As linhas de base de Davis usam descritores de composição de aminoácidos no lado proteico, não ESM2, porque o ambiente de execução não tinha acesso ao repositório de pesos. É deliberadamente uma linha de base.

O experimento kekulizado não terminou.

## Segurança do código

O arquivo de rótulos do Davis é um pickle baixado da internet, e desempacotar pickle de fonte não confiável executa código arbitrário. O script usa um desempacotador restrito que só permite classes do numpy.

## Referências

nanoGPT, de Andrej Karpathy, base da implementação do transformer.

MolGPT, Bagal et al., Journal of Chemical Information and Modeling, 2021, geração de moléculas condicionada com transformer.

ZINC250k, na versão distribuída pelo repositório chemical_vae do grupo Aspuru Guzik.

Davis et al., Nature Biotechnology, 2011, para o conjunto de afinidade de quinases, usado através do espelho do DeepDTA.

DeepDTA, Öztürk et al., Bioinformatics, 2018, referência de comparação em Davis.

RDKit, toolkit de quimioinformática.

MOSES, Polykovskiy et al., e GuacaMol, Brown et al., para as métricas padrão de modelo generativo molecular.

## Licença

MIT.
