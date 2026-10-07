# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
  <a href="https://www.fiap.com.br/">
    <img src="https://github.com/Luiz-Frederico/templateFiap/blob/main/assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Administração Paulista" border="0" width="40%" height="40%">
  </a>
</p>

<br>

---

# 🫀 CardioIA — Processamento de Sintomas e Classificação de ECG com Inteligência Artificial

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-3776AB?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

## Integrante

<p align="left">
  <a href="https://github.com/Luiz-Frederico" target="_blank">
    <img src="https://github.com/Luiz-Frederico.png" width="64" height="64" alt="@Luiz-Frederico" />
  </a>
</p>

**Luiz Frederico Nunes Campelo**

## Professores

### Coordenador(a) / Tutor(a)

<p align="left">
  <a href="https://github.com/agodoi" target="_blank">
    <img src="https://github.com/agodoi.png" width="64" height="64" alt="@agodoi" />
  </a>
  <a href="https://github.com/SabrinaOtoni" target="_blank">
    <img src="https://github.com/SabrinaOtoni.png" width="64" height="64" alt="@SabrinaOtoni" />
  </a>
</p>

---

## 📜 Descrição

O **CardioIA** é um projeto acadêmico desenvolvido na Fase 2 com o objetivo de aplicar técnicas de **Processamento de Linguagem Natural, Machine Learning e Redes Neurais** em um cenário experimental de apoio à triagem de risco cardiovascular.

A solução foi organizada em três frentes complementares.

Na **Parte 1**, foi construído um mapa de conhecimento que relaciona pares de sintomas a possíveis condições cardíacas. O sistema processa relatos textuais, normaliza expressões, utiliza lematização com spaCy e identifica associações presentes na ontologia. Também foi incorporado tratamento de negação por análise de dependência sintática, evitando que sintomas explicitamente negados contribuam para a sugestão produzida pelo sistema.

Na **Parte 2**, foi desenvolvido um classificador supervisionado para distinguir relatos de **alto risco** e **baixo risco**. Os textos são pré-processados com a mesma lógica de negação utilizada na Parte 1, representados numericamente com **TF-IDF** e classificados por **Regressão Logística**. O desenvolvimento incluiu análise de pesos, auditoria de erros, investigação de um falso positivo, ampliação do dataset e reavaliação do modelo.

No **Ir Além 2**, o projeto foi ampliado para visão computacional, utilizando uma **MLP (Perceptron Multicamadas)** implementada com TensorFlow/Keras para classificar imagens de eletrocardiogramas em **Normal** ou **Anormal**. O fluxo contemplou auditoria do dataset, pré-processamento das imagens, comparação controlada de arquiteturas e hiperparâmetros, seleção do modelo e avaliação final em conjunto de teste reservado.

O projeto possui finalidade exclusivamente **acadêmica e experimental**. As associações, classificações e previsões produzidas não representam diagnóstico médico e não devem substituir avaliação de profissionais de saúde.

---

## 🔎 Visão Geral

| Componente | Objetivo | Abordagem principal | Resultado |
|---|---|---|---|
| **Parte 1** | Extrair sintomas e relacioná-los a possíveis condições | spaCy + mapa de conhecimento + regras | 14 relatos processados e 15 associações |
| **Parte 2** | Classificar relatos em alto ou baixo risco | TF-IDF + Regressão Logística | **100%** no conjunto de teste controlado |
| **Ir Além 2** | Classificar imagens de ECG em Normal ou Anormal | MLP com TensorFlow/Keras | **84,36%** no teste final |

---

## ⭐ Principais Entregáveis

O projeto contempla:

- processamento automatizado de relatos de sintomas;
- mapa de conhecimento em CSV com associações entre sintomas e condições;
- normalização, lematização e tratamento de negação com spaCy;
- exportação estruturada dos resultados da Parte 1;
- dataset textual balanceado para classificação de risco;
- vetorização de textos com TF-IDF;
- classificador com Regressão Logística;
- análise dos pesos aprendidos pelo modelo;
- auditoria e investigação de erros de classificação;
- teste interativo com novas frases;
- auditoria e pré-processamento de imagens de ECG;
- MLP com TensorFlow/Keras;
- experimentos controlados de arquitetura e treinamento;
- avaliação final com conjunto de teste reservado;
- discussão sobre IA responsável, limitações e eficiência computacional.

---

# 🧠 Parte 1 — Extração de Sintomas e Mapa de Conhecimento

## Objetivo

A Parte 1 foi desenvolvida para processar pequenos relatos de pacientes, identificar expressões relacionadas a sintomas e compará-las com uma ontologia criada para a atividade.

O fluxo permite:

```text
Relato do paciente
        ↓
Normalização textual
        ↓
Lematização com spaCy
        ↓
Detecção do escopo de negação
        ↓
Identificação dos sintomas
        ↓
Consulta ao mapa de conhecimento
        ↓
Pontuação das associações
        ↓
Sugestão simulada de condição
        ↓
Exportação dos resultados
```

O sistema utiliza uma lógica de pontuação na qual uma associação completa entre dois elementos possui maior peso do que uma correspondência parcial. Essa decisão reduz o risco de um único sintoma isolado gerar automaticamente múltiplas condições.

---

## Mapa de Conhecimento

A ontologia contém **15 associações** distribuídas entre quatro condições:

| Condição | Associações |
|---|---:|
| Infarto | 6 |
| Insuficiência Cardíaca | 3 |
| Angina | 4 |
| Arritmia Cardíaca | 2 |
| **Total** | **15** |

Os dados foram armazenados em arquivo CSV, separando o conhecimento utilizado da lógica de processamento.

---

## Tratamento de Negação

Além do matching textual e lematizado, foi implementada detecção de negação utilizando a árvore de dependências sintáticas do spaCy.

Expressões como:

```text
não
nunca
jamais
nem
tampouco
```

são utilizadas como indicadores de negação.

Quando um sintoma é identificado dentro do escopo de uma negação, ele é marcado como:

```text
(NEGADO)
```

e não recebe pontuação para a sugestão de condição.

Exemplo:

```text
Relato:
Não sinto dor no peito, apenas um leve cansaço nas pernas.

Sintoma identificado:
dor no peito (NEGADO)

Sugestão:
Diagnóstico Indeterminado
```

Essa abordagem evita que a simples presença textual de uma expressão como `dor no peito` seja interpretada como evidência positiva quando o próprio relato afirma que o sintoma não está presente.

---

## 📊 Resultados da Parte 1

Foram utilizados **10 relatos-base** previstos na atividade e acrescentados **4 casos complementares** para validar o tratamento de negação.

| Indicador | Resultado |
|---|---:|
| Relatos-base | **10** |
| Casos complementares de negação | **4** |
| Total de relatos processados | **14** |
| Associações no mapa de conhecimento | **15** |
| Relatos com sintomas identificados | **14** |
| Relatos com condição sugerida | **10** |
| Casos de negação com resultado indeterminado | **4** |

### Distribuição dos resultados

| Resultado sugerido | Quantidade |
|---|---:|
| Infarto | **4** |
| Insuficiência Cardíaca | **3** |
| Angina | **2** |
| Arritmia Cardíaca | **1** |
| Diagnóstico Indeterminado | **4** |
| **Total** | **14** |

Os quatro relatos criados especificamente para validar negação resultaram em **Diagnóstico Indeterminado**, pois os sintomas relevantes foram identificados como negados e não participaram da pontuação.

### Exemplos de resultados

A seguir, alguns exemplos reais produzidos pelo processamento da Parte 1:

#### Infarto

```text
Relato:
Há dois dias estou com uma dor no peito intensa que piora quando faço esforço físico.

Sintomas identificados:
dor no peito, esforço físico

Diagnóstico sugerido:
Infarto

Pontuação:
3

Associação encontrada:
dor no peito + esforço físico → Infarto
```

#### Insuficiência Cardíaca

```text
Relato:
Sinto cansaço constante há uma semana, mesmo depois de descansar o dia todo.

Sintomas identificados:
cansaço constante, mesmo depois de descansar

Diagnóstico sugerido:
Insuficiência Cardíaca

Pontuação:
3

Associação encontrada:
cansaço constante + mesmo depois de descansar → Insuficiência Cardíaca
```

#### Angina

```text
Relato:
Estou com uma falta de ar muito forte e dificuldade para respirar desde ontem à noite.

Sintomas identificados:
dificuldade para respirar, falta de ar

Diagnóstico sugerido:
Angina

Pontuação:
3

Associação encontrada:
falta de ar + dificuldade para respirar → Angina
```

#### Arritmia Cardíaca

```text
Relato:
Estou sentindo palpitações no coração e uma dor no peito leve que vai e volta.

Sintomas identificados:
dor no peito, palpitações

Diagnóstico sugerido:
Arritmia Cardíaca

Pontuação:
3

Associação encontrada:
palpitações + dor no peito → Arritmia Cardíaca
```

#### Exemplo com negação

```text
Relato:
Não tenho falta de ar nem palpitações, somente um desconforto muscular.

Sintomas identificados:
falta de ar (NEGADO), palpitações (NEGADO)

Diagnóstico sugerido:
Diagnóstico Indeterminado

Pontuação:
0

Associação encontrada:
Nenhuma associação completa encontrada
```

Esses exemplos demonstram tanto o funcionamento das associações completas do mapa de conhecimento quanto o tratamento de sintomas explicitamente negados.

---

## Entregáveis da Parte 1

```text
sintomas_pacientes.txt
ontologia_sintomas.csv
resultados_diagnostico_parte1.csv
```

Além dos arquivos, foram implementados:

- normalização textual;
- lematização;
- extração de sintomas;
- detecção de negação;
- pontuação por associação;
- exportação dos resultados em DataFrame/CSV.

---

# 🤖 Parte 2 — Classificador de Risco Cardiovascular

## Objetivo

A Parte 2 utiliza Machine Learning supervisionado para classificar relatos textuais em:

```text
alto risco
baixo risco
```

O fluxo utilizado foi:

```text
Dataset rotulado
      ↓
Tratamento de negação
      ↓
Divisão treino/teste
      ↓
TF-IDF
      ↓
Regressão Logística
      ↓
Predição
      ↓
Avaliação
      ↓
Auditoria de erros
      ↓
Ajuste do dataset
      ↓
Retreinamento
```

---

## Integração com a Parte 1

A lógica de negação implementada na Parte 1 foi reaproveitada como pré-processamento na Parte 2.

Tokens sob escopo de negação recebem o prefixo:

```text
NEG_
```

Assim:

```text
dor
```

e

```text
NEG_dor
```

passam a ser características diferentes para o TF-IDF.

Isso permite ao modelo distinguir, por exemplo:

```text
"Sinto dor no peito"
```

de:

```text
"Não sinto dor no peito"
```

O mesmo pré-processamento é aplicado durante treino, teste e inferência.

---

## TF-IDF e Regressão Logística

As frases são transformadas em vetores utilizando:

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True,
    sublinear_tf=True
)
```

Em seguida, os vetores são utilizados por um modelo de:

```text
Regressão Logística
```

A divisão dos dados utiliza `random_state=42` e estratificação das classes.

---

## Auditoria do Modelo

A primeira versão do dataset continha **50 frases**, igualmente divididas entre alto e baixo risco.

```text
50 frases
├── 25 alto risco
└── 25 baixo risco
```

Com divisão 80/20:

```text
40 frases → treino
10 frases → teste
```

O primeiro modelo alcançou:

> **90,00% de acurácia**

A matriz de confusão revelou um falso positivo:

```text
Real: baixo risco
Previsto: alto risco
```

O caso foi investigado individualmente em vez de simplesmente alterar hiperparâmetros.

Também foram analisados os pesos atribuídos pela Regressão Logística aos termos gerados pelo TF-IDF, permitindo observar quais palavras e expressões influenciavam a decisão do classificador.

---

## Ajuste do Dataset

A auditoria indicou a necessidade de aumentar a variedade de contextos de expressões como:

```text
dor
cansaço
falta de ar
```

Foram adicionadas **10 novas frases contextualizadas**, mantendo o balanceamento.

O dataset final ficou com:

```text
60 frases
├── 30 alto risco
└── 30 baixo risco
```

A divisão final 80/20 resultou em:

```text
48 frases → treino
12 frases → teste
```

---

## 📊 Resultados da Parte 2

Após o ajuste do dataset e o retreinamento:

| Métrica | Resultado |
|---|---:|
| Dataset final | **60 frases** |
| Alto risco | **30** |
| Baixo risco | **30** |
| Treino | **48 frases** |
| Teste | **12 frases** |
| Acurácia anterior | **90,00%** |
| Acurácia final | **100,00%** |
| Melhoria | **+10,00 p.p.** |
| Falsos positivos | **0** |
| Falsos negativos | **0** |

### Matriz de confusão final

```text
[[6 0]
 [0 6]]
```

### Relatório de classificação

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Alto risco | **1,00** | **1,00** | **1,00** | 6 |
| Baixo risco | **1,00** | **1,00** | **1,00** | 6 |
| **Accuracy** |  |  | **1,00** | **12** |

O resultado de 100% ocorreu em um conjunto **pequeno, simulado e controlado**. Portanto, não representa desempenho esperado em dados clínicos reais.

---

## Teste Interativo

O notebook também permite inserir um novo relato de até 100 caracteres.

A frase é processada pela mesma função de tratamento de negação utilizada no treinamento antes de ser vetorizada.

Exemplo executado:

```text
Relato:
Não sinto dor no peito, apenas um cansaço leve

Classificação:
BAIXO RISCO

Confiança:
67,3%
```

Essa funcionalidade serve apenas para demonstração acadêmica do pipeline.

---

## Entregáveis da Parte 2

```text
base_triagem_risco.csv
classificador TF-IDF + Regressão Logística
tratamento de negação integrado
análise de pesos
auditoria de erros
ajuste do dataset
matriz de confusão
classification report
teste interativo
```

---

# ❤️ Ir Além 2 — Diagnóstico Visual em Cardiologia com Rede Neural

## Objetivo

O Ir Além 2 amplia o CardioIA para classificação visual de eletrocardiogramas.

O objetivo é distinguir imagens classificadas como:

```text
0 → Normal
1 → Anormal
```

utilizando uma **MLP (Perceptron Multicamadas)** implementada com TensorFlow/Keras.

> O modelo é apresentado exclusivamente como recurso acadêmico e experimental de apoio à triagem.

---

## Dataset de ECG

Foi utilizado o dataset público:

**ECG Image Dataset (Normal/Abnormal)**  
Kaggle:

https://www.kaggle.com/datasets/y20cs3255ramu/ecg-image-dataset-normal-abnormal

### Auditoria

| Característica | Resultado |
|---|---:|
| Total de imagens | **1.405** |
| Normal | **859 (61,14%)** |
| Anormal | **546 (38,86%)** |
| Dimensão original | **2105 × 1190** |
| Formato original | RGB |
| Arquivos problemáticos | **0** |

A distribuição apresenta desbalanceamento moderado, com maior quantidade de imagens classificadas como normais.

---

## Pré-processamento das Imagens

O pipeline definitivo foi:

```text
RGB
 ↓
Escala de cinza
 ↓
128 × 128 pixels
 ↓
Normalização [0,1]
 ↓
MLP
```

Também foi realizada uma avaliação visual de um filtro leve de nitidez. Como o ganho observado foi pequeno e poderia reforçar a grade presente nas imagens, nenhum filtro adicional foi incorporado ao pipeline definitivo.

---

## Separação dos Dados

Foi utilizada divisão estratificada com `random_state=42`.

| Conjunto | Total | Normal | Anormal |
|---|---:|---:|---:|
| Treino | **983** | 601 | 382 |
| Validação | **211** | 129 | 82 |
| Teste | **211** | 129 | 82 |

Proporção:

```text
70% treino
15% validação
15% teste
```

O conjunto de teste permaneceu reservado durante a escolha da arquitetura e dos hiperparâmetros.

---

## Arquitetura da MLP

A arquitetura selecionada foi:

```text
Input 128 × 128
        ↓
Flatten
        ↓
Dense 128 — ReLU
        ↓
Dense 64 — ReLU
        ↓
Dense 1 — Sigmoid
```

Total:

> **2.105.601 parâmetros treináveis**

---

## Experimentos Controlados

Inicialmente, o SGD foi avaliado com diferentes taxas de aprendizagem.

A redução do `learning_rate` para **0,001** tornou a evolução das métricas mais gradual e estável.

### Comparação de arquiteturas com SGD 0,001

| Arquitetura | Parâmetros | Melhor época | Val. Accuracy | Val. Loss |
|---|---:|---:|---:|---:|
| 64–32 | 1.050.753 | 124 | 80,09% | 0,5154 |
| **128–64** | **2.105.601** | **126** | **82,46%** | 0,5129 |
| 256–128 | 4.227.585 | 139 | 81,04% | **0,5075** |

A arquitetura **128–64** apresentou a maior acurácia de validação nessa comparação, enquanto a arquitetura 256–128 praticamente dobrou o número de parâmetros sem melhorar a métrica principal.

---

## Seleção com Adam

Mantendo:

```text
Resolução:       128 × 128
Arquitetura:     128–64
Batch size:      32
Seed:            42
Loss:            Binary Crossentropy
```

o otimizador foi alterado para:

```text
Adam
learning rate = 0,0005
```

O melhor resultado de validação foi:

| Métrica | Resultado |
|---|---:|
| Melhor época | **137** |
| Acurácia de treino | **85,35%** |
| Acurácia de validação | **84,83%** |
| Validation loss | **0,3752** |
| Diferença treino × validação | **0,52 p.p.** |

As curvas apresentaram oscilações ao longo do treinamento, mas o modelo atingiu um ponto de boa generalização na época selecionada.

Configurações adicionais de regularização e treinamento foram exploradas durante o desenvolvimento, mas não produziram benefício suficiente para substituir a configuração selecionada.

---

## 🏆 Modelo Final

A configuração final foi:

| Parâmetro | Configuração |
|---|---|
| Resolução | **128 × 128** |
| Arquitetura | **128–64** |
| Otimizador | **Adam** |
| Learning rate | **0,0005** |
| Batch size | **32** |
| Seed | **42** |
| Épocas finais | **137** |
| Threshold | **0,50** |
| Parâmetros | **2.105.601** |

O modelo foi reconstruído com a mesma configuração e treinado por **137 épocas** antes da abertura do conjunto de teste.

---

## 📊 Resultados do Ir Além 2

| Métrica | Resultado |
|---|---:|
| Imagens de teste | **211** |
| Test loss | **0,3394** |
| Melhor acurácia de validação | **84,83%** |
| **Acurácia final de teste** | **84,36%** |
| Diferença validação × teste | **0,47 p.p.** |

A proximidade entre validação e teste indica comportamento consistente nesta divisão experimental.

### Matriz de confusão

```text
[[115  14]
 [ 19  63]]
```

Considerando:

```text
0 = Normal
1 = Anormal
```

o modelo apresentou:

| Resultado | Quantidade |
|---|---:|
| Normal classificado corretamente | **115** |
| Normal classificado como Anormal | **14** |
| Anormal classificado como Normal | **19** |
| Anormal classificado corretamente | **63** |
| Classificações corretas | **178** |
| Classificações incorretas | **33** |

---

# 📊 Resumo Geral dos Resultados

| Componente | Base avaliada | Resultado principal |
|---|---:|---:|
| Parte 1 | 14 relatos | 10 sugestões + 4 casos de negação indeterminados |
| Parte 1 | 15 associações | 4 condições representadas |
| Parte 2 | 60 frases | Dataset final 30/30 |
| Parte 2 | 12 exemplos de teste | **100,00% de acurácia** |
| Ir Além 2 | 1.405 imagens | Dataset ECG auditado |
| Ir Além 2 | 211 imagens de validação | **84,83%** |
| Ir Além 2 | 211 imagens de teste | **84,36%** |
| Ir Além 2 | Modelo final | **2.105.601 parâmetros** |

---

# 🛡️ IA Responsável

Como o projeto está relacionado à saúde, as soluções desenvolvidas devem ser interpretadas exclusivamente como demonstrações acadêmicas.

Os módulos não representam:

- diagnóstico médico;
- recomendação terapêutica;
- dispositivo médico validado;
- sistema autônomo de decisão clínica.

Na Parte 2, a auditoria dos pesos e a investigação individual dos erros foram utilizadas para compreender o comportamento do classificador antes de alterar o modelo.

Também foram documentadas limitações de representatividade. Os datasets utilizados não possuem informações suficientes para medir desempenho entre diferentes grupos demográficos ou clínicos.

No Ir Além 2, a acurácia final de **84,36%** implica a existência de erros de classificação, incluindo imagens anormais classificadas como normais. Por isso, supervisão humana seria indispensável em qualquer cenário real.

O projeto não realiza validação clínica.

---

# 🌱 Sustentabilidade e Eficiência Computacional

A eficiência computacional foi considerada principalmente no Ir Além 2.

As imagens originais RGB foram convertidas para escala de cinza e reduzidas para **128 × 128 pixels**, diminuindo a dimensionalidade de entrada.

Na comparação de arquiteturas:

```text
128–64  → 2.105.601 parâmetros
256–128 → 4.227.585 parâmetros
```

A arquitetura maior praticamente dobrou o número de parâmetros sem melhorar a acurácia de validação.

Por isso, a arquitetura 128–64 foi mantida, evitando complexidade adicional sem benefício mensurável.

Não foram realizadas medições de energia ou emissões de carbono. Portanto, as considerações de sustentabilidade estão limitadas à **eficiência e complexidade computacional**.

---

# ⚠️ Limitações

Entre as principais limitações do projeto estão:

- os dados textuais das Partes 1 e 2 são simulados e controlados;
- o resultado de 100% da Parte 2 foi obtido em apenas 12 amostras de teste;
- a ontologia da Parte 1 foi construída especificamente para o escopo acadêmico da atividade;
- TF-IDF não compreende contexto semântico de forma nativa, exigindo tratamento explícito de negação;
- os datasets não possuem metadados demográficos suficientes para análise de fairness;
- o dataset de ECG contém apenas 1.405 imagens;
- existe desbalanceamento moderado entre as classes do dataset de ECG;
- o redimensionamento para 128 × 128 pode reduzir detalhes da imagem original;
- a MLP utiliza `Flatten` e não explora diretamente a estrutura espacial das imagens;
- a acurácia não descreve isoladamente todos os tipos de erro relevantes em saúde;
- não foi realizada validação externa;
- não foi realizada validação clínica;
- os resultados não podem ser generalizados para utilização médica real.

---

# 📁 Estrutura de Pastas

A estrutura utilizada no projeto pode ser organizada da seguinte forma:

```text
CardioIA/
│
├── notebooks/
│   ├── FASE_2_Diagnostico_Automatizado_IA_no_Estetoscopio_Digital.ipynb
│   └── Ir_Alem_2_CardioIA_MLP_ECG.ipynb
│
├── data/
│   ├── sintomas_pacientes.txt
│   ├── ontologia_sintomas.csv
│   ├── resultados_diagnostico_parte1.csv
│   └── base_triagem_risco.csv
│
├── images/
│   └── Crop_Dataset/
│       ├── ECG_Abnormal/
│       └── ECG_Normal/
│
├── assets/
│   └── logo-fiap.png
│
└── README.md
```

A pasta `images/` é utilizada pelo **Ir Além 2** para armazenar o dataset de ECG.

Os arquivos `.txt` e `.csv` das Partes 1 e 2 podem ser gerados diretamente pelo notebook durante a execução.

---

# 🔧 Como Executar o Código

## Pré-requisitos

Recomenda-se utilizar:

- Python 3.10 ou superior;
- Google Colab ou ambiente compatível com Jupyter Notebook;
- acesso à internet para instalação/download inicial de dependências e dataset.

### Bibliotecas principais

```bash
pip install pandas numpy scikit-learn spacy tensorflow matplotlib pillow kagglehub
python -m spacy download pt_core_news_sm
```

---

## Parte 1 e Parte 2

Abra o notebook:

```text
FASE_2_Diagnostico_Automatizado_IA_no_Estetoscopio_Digital.ipynb
```

Execute as células na ordem apresentada.

O notebook realiza:

```text
Instalação/carregamento do spaCy
        ↓
Criação dos relatos
        ↓
Criação da ontologia
        ↓
Extração de sintomas
        ↓
Tratamento de negação
        ↓
Exportação dos resultados
        ↓
Criação do dataset de triagem
        ↓
TF-IDF
        ↓
Regressão Logística
        ↓
Auditoria
        ↓
Ajuste do dataset
        ↓
Reavaliação
        ↓
Teste interativo
```

---

## Ir Além 2

O dataset de imagens de ECG é obtido diretamente do Kaggle utilizando a biblioteca `kagglehub`. Portanto, não é necessário baixar manualmente as imagens ou adicioná-las ao repositório.

### Instalação do KaggleHub

```python
!pip install -q kagglehub
```
```
import kagglehub

# Download da versão mais recente do dataset
path = kagglehub.dataset_download(
    "y20cs3255ramu/ecg-image-dataset-normal-abnormal"
)

print("Dataset baixado em:")
print(path)
```
O kagglehub.dataset_download() retorna o caminho local em que o dataset foi armazenado no ambiente de execução.
A partir desse caminho, o notebook acessa as duas classes disponíveis no dataset:

```
Crop_Dataset/
├── ECG_Abnormal/
└── ECG_Normal/
```

Em seguida, abra:

```text
Ir_Alem_2_CardioIA_MLP_ECG.ipynb
```

e execute as células na ordem apresentada.

O notebook realiza:

```Download do dataset
        ↓
Auditoria das imagens
        ↓
Análise visual
        ↓
Pré-processamento
        ↓
Divisão treino / validação / teste
        ↓
Construção da MLP
        ↓
Treinamento
        ↓
Experimentos controlados
        ↓
Seleção do modelo
        ↓
Avaliação final no conjunto de teste
```

---

# 🛠️ Tecnologias e Bibliotecas Utilizadas

### Linguagem

- Python

### Dados

- Pandas
- NumPy

### Processamento de Linguagem Natural

- spaCy
- expressões regulares
- normalização textual
- análise de dependência sintática

### Machine Learning

- scikit-learn
- TF-IDF
- Regressão Logística

### Deep Learning

- TensorFlow
- Keras
- MLP

### Imagens e Visualização

- Pillow (PIL)
- Matplotlib

### Ambiente e Dados Externos

- Google Colab
- Kaggle
- KaggleHub

---

# 🎥 Vídeo de Apresentação

Vídeo de apresentação da Fase 2 do projeto CardioIA:

**YouTube:**  
[LINK_DO_VIDEO_YOUTUBE]

---

# 🔗 Repositório

**GitHub:**  
[URL_DO_REPOSITORIO]

---

# 📚 Referências

### Material FIAP

- **Cap02 — RPA na Veia: Automatizando Arquivos, Redes e Sistemas com Python**
- **Cap06 — IA Criativa: Desvendando o Cérebro das Máquinas que Criam**
- **Cap07 — IA Responsável: Ética, Sustentabilidade e Regulação na Era dos Dados**
- **Cap10 — IA que Entende: Processamento de Linguagem Natural Baseado em Regras**
- **Cap11 — NLP no Estilo Clássico: Estatística, Vetores e Emoções em Texto**
- **Cap12 — Máquinas que Enxergam: Filtragem Inteligente de Imagens com IA**

### Dataset

- **ECG Image Dataset (Normal/Abnormal)** — Kaggle  
  https://www.kaggle.com/datasets/y20cs3255ramu/ecg-image-dataset-normal-abnormal

### Documentações

- Python
- pandas
- NumPy
- spaCy
- scikit-learn
- TensorFlow/Keras
- Matplotlib
- Pillow
- Kaggle/KaggleHub

---

# 🗃 Histórico de Lançamentos

- **0.3.0 - 07/10/2026**
  - Integração do tratamento de negação entre as Partes 1 e 2.
  - Ampliação da base textual para 60 frases balanceadas.
  - Auditoria e reavaliação do classificador.
  - Consolidação do Ir Além 2 com MLP para imagens de ECG.
  - Seleção do modelo final com Adam.
  - Avaliação final de **84,36%** no conjunto de teste de ECG.
  - Consolidação da documentação final do projeto.

- **0.2.0 - 03/10/2026**
  - Implementação do classificador de risco cardiovascular.
  - Aplicação de TF-IDF.
  - Implementação da Regressão Logística.
  - Avaliação e auditoria inicial do modelo.

- **0.1.0 - 03/10/2026**
  - Criação dos relatos de sintomas.
  - Criação do mapa de conhecimento.
  - Implementação da extração de sintomas.
  - Associação entre sintomas e possíveis condições.

---

# ✅ Conclusão

O CardioIA permitiu explorar três abordagens complementares de Inteligência Artificial aplicadas a um contexto acadêmico de apoio à triagem cardiovascular.

Na **Parte 1**, foi desenvolvido um sistema baseado em regras e mapa de conhecimento para extrair sintomas de relatos textuais, incluindo tratamento explícito de negação por meio da análise de dependência sintática do spaCy.

Na **Parte 2**, a mesma lógica de negação foi integrada a um pipeline clássico de Machine Learning utilizando **TF-IDF e Regressão Logística**. A auditoria dos erros levou à ampliação do dataset para **60 frases balanceadas**, e o modelo alcançou **100% de acurácia nas 12 amostras do conjunto de teste controlado**.

No **Ir Além 2**, foi desenvolvida uma MLP em Keras para classificação de imagens de ECG. Após experimentos controlados, a configuração com arquitetura **128–64**, otimizador **Adam**, learning rate **0,0005**, batch size 32 e seed 42 apresentou a melhor combinação avaliada.

A melhor acurácia de validação foi de **84,83%**, enquanto a avaliação final sobre **211 imagens mantidas fora do processo de seleção** alcançou **84,36%**, com loss de **0,3394**.

Os resultados demonstram a viabilidade experimental das técnicas utilizadas, mas devem ser interpretados dentro das limitações dos datasets, dos modelos e da ausência de validação clínica.

---

> ⚠️ **Uso exclusivamente acadêmico.**
>
> As associações, classificações e previsões deste projeto não representam diagnóstico médico e não devem ser utilizadas para tomada de decisão clínica.

---

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1">

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/agodoi/template">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">FIAP</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>

---
