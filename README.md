# amigos
amigos unidos contra machine learning

# Regressão Ridge e Lasso

Os modelos de regressão Ridge e Lasso são técnicas de regressão linear regularizada que ajudam a prevenir overfitting (superajuste) em modelos, especialmente quando você está lidando com muitos recursos (features) e/ou um número limitado de observações. Ambas as técnicas introduzem uma penalização (regularização) na função de custo, mas fazem isso de maneiras diferentes.

- Quando usar *Ridge*
1. Quando você tem muitos recursos que são correlacionados entre si (multicolinearidade).
2. Quando você deseja reduzir a variância do modelo sem eliminar completamente os coeficientes (todos os recursos são mantidos no modelo, mas com coeficientes reduzidos).
3. Ideal para quando você acredita que todos os recursos têm algum impacto, mas não necessariamente com alta magnitude.
- Quando usar *Lasso*
1. Quando você deseja um modelo mais simples e interpretável, eliminando recursos irrelevantes (feature selection).
2. Ideal quando você suspeita que apenas um subconjunto pequeno de recursos seja realmente útil para prever a variável de interesse.
3. Útil em situações de alta dimensionalidade, onde o número de recursos é próximo ou maior que o número de observações


# Processo de Escolha de Modelos em Machine Learning

O processo de escolha de modelos em machine learning é crucial para garantir que o modelo selecionado seja o mais apropriado para o problema em questão. As etapas fundamentais a seguir detalham esse processo:

## X_validação Y_validação

### 1. Separação dos Dados

O primeiro passo consiste em dividir a base de dados original, que contém as features (variáveis preditoras) e a variável alvo (target), em conjuntos de treinamento e teste:

- **X e Y**: Representam os conjuntos de dados originais, onde X contém as features e Y a variável alvo.
- **Conjunto de Treinamento (X_treino, Y_treino)**: Geralmente, 70-80% dos dados é reservado para treinar o modelo.
- **Conjunto de Teste (X_teste, Y_teste)**: Os 20-30% restantes são utilizados para avaliar o desempenho final do modelo.

Essa separação inicial assegura que os dados de teste sejam utilizados apenas na fase de avaliação, simulando como o modelo se comportará com dados novos.

### 2. Divisão do Treino em Treino-Validação

Dentro do conjunto de treinamento, é comum realizar uma nova divisão:

- **Conjunto de Treino-Validação (X_treino_validação, Y_treino_validação)**: Usado para treinar o modelo.
- **Conjunto de Validação (X_validação, Y_validação)**: Usado para validar o desempenho do modelo durante o ajuste.

Essa divisão é essencial para garantir que a avaliação do modelo seja realizada em dados que não foram vistos durante o treinamento, ajudando a prevenir overfitting, que ocorre quando o modelo se ajusta excessivamente aos dados de treino e perde a capacidade de generalização.

### 3. Escolha do Modelo

Com os conjuntos de treino-validação e validação definidos, diferentes modelos e combinações de hiperparâmetros são testados. O desempenho é avaliado com base em métricas de erro, como o RMSE (Root Mean Squared Error), utilizando os dados de validação.

- **Ajuste de Hiperparâmetros**: Técnicas como Grid Search e Random Search são utilizadas para identificar a melhor combinação de hiperparâmetros.
- **Comparação de Modelos**: Diversos modelos (como regressão linear, árvore de decisão, etc.) são treinados e avaliados, e o modelo com melhor desempenho em termos de RMSE no conjunto de validação é selecionado.
- **O modelo com melhor desempenho é escolhido e aplicado no X_test Y_test**

### Resumo do Processo

1. **Divisão Inicial**: X e Y são separados em conjuntos de treino e teste.
2. **Divisão do Treino**: O conjunto de treino é dividido em treino-validação e validação.
3. **Ajuste de Modelos e Hiperparâmetros**: Diversos modelos são treinados e avaliados usando o conjunto de validação. Para cada modelo, utilizamos o **X_treino_validação** e o **Y_treino_validação** para ajustar os parâmetros e então aplicamos o modelo no **X_validação** e **Y_validação**. Com base no desempenho em **Y_validação** escolhemos o melhor modelo que será aplicado em *X_test** e **Y_test**.


## Validação Cruzada



### 1. Separação dos Dados

O primeiro passo consiste em dividir a base de dados original, que contém as features (variáveis preditoras) e a variável alvo (target), em conjuntos de treinamento e teste:

- **X e Y**: Representam os conjuntos de dados originais, onde X contém as features e Y a variável alvo.
- **Conjunto de Treinamento (X_treino, Y_treino)**: Dividimos em **X_treino**, **Y_treino** e **X_teste**, **Y_test**

### 2. Validação Cruzada (Cross-Validation)

A validação cruzada é uma técnica que aprimora a robustez da avaliação dos modelos. O conjunto de treino é dividido em "k" subconjuntos (folds), permitindo que o modelo seja treinado e avaliado "k" vezes, cada vez usando um subconjunto diferente para teste e os demais para treino.

- **k-Fold Cross-Validation**: Por exemplo, ao escolher `k=5`, o conjunto de treino é segmentado em 5 partes. Em cada iteração, uma parte é utilizada como conjunto de validação, enquanto as outras quatro partes são usadas para treinar o modelo. Esse processo é repetido 5 vezes, e a média das métricas de erro é calculada, resultando em uma avaliação geral do modelo.
- **Vantagens**: Essa abordagem permite uma estimativa mais estável do desempenho do modelo, utilizando todos os dados de treino para avaliação.

### 3. Avaliação Final com Dados de Teste

Após a seleção do melhor modelo, ele é ajustado com o conjunto completo de treino (X_treino, Y_treino) e, em seguida, avaliado no conjunto de teste (X_teste, Y_teste). O desempenho do modelo neste conjunto é utilizado para estimar sua capacidade de generalização. Métricas como o RMSE são calculadas para fornecer uma medida quantitativa do erro do modelo.

### Resumo do Processo

1. **Divisão Inicial**: X e Y são separados em conjuntos de treino e teste.
2. **Validação Cruzada**: A técnica de k-fold é aplicada para uma avaliação mais robusta.
3. **Avaliação Final**: O modelo final é avaliado nos dados de teste, utilizando métricas como RMSE para medir a performance.

Esse processo sistemático assegura que o modelo escolhido seja capaz de generalizar adequadamente para novos dados, minimizando os riscos de overfitting e subfitting.

# TP, FP, TN e FN

## 1. Verdadeiros Positivos (TP), Falsos Positivos (FP), Verdadeiros Negativos (TN), Falsos Negativos (FN)

- **Verdadeiros Positivos (TP)**: Número de instâncias positivas corretamente classificadas pelo modelo.
- **Falsos Positivos (FP)**: Número de instâncias negativas incorretamente classificadas como positivas.
- **Verdadeiros Negativos (TN)**: Número de instâncias negativas corretamente classificadas pelo modelo.
- **Falsos Negativos (FN)**: Número de instâncias positivas incorretamente classificadas como negativas.

## 2. Precisão (Precision)

A precisão é a proporção de verdadeiros positivos em relação ao total de instâncias classificadas como positivas.

**Fórmula**: Precision = TP / (TP + FP)

**Exemplo**: Se um modelo classifica 80 instâncias como positivas, das quais 70 são verdadeiros positivos e 10 são falsos positivos, a precisão será:

**Cálculo**: Precision = 70 / (70 + 10) = 0.88 (88%)

## 3. Recall (Sensibilidade)

O recall é a proporção de verdadeiros positivos em relação ao total de instâncias realmente positivas.

**Fórmula**: Recall = TP / (TP + FN)

**Exemplo**: Se existem 100 instâncias positivas e o modelo identifica corretamente 70 delas, o recall será:

**Cálculo**: Recall = 70 / (70 + 30) = 0.70 (70%)

## 4. Especificidade (Specificity)

A especificidade é a proporção de verdadeiros negativos em relação ao total de instâncias realmente negativas.

**Fórmula**: Especificidade = TN / (TN + FP)

**Exemplo**: Se existem 90 instâncias negativas e o modelo identifica corretamente 80 delas, a especificidade será:

**Cálculo**: Especificidade = 80 / (80 + 10) = 0.89 (89%)

## 5. TNR (Taxa de Verdadeiros Negativos)

A TNR é sinônimo de especificidade e representa a mesma métrica.

## 6. Curva ROC (Receiver Operating Characteristic)

A Curva ROC é um gráfico que ilustra o desempenho de um modelo de classificação em diferentes limiares de decisão. Ela plota a taxa de verdadeiros positivos (TPR) contra a taxa de falsos positivos (FPR).

- **Taxa de Verdadeiros Positivos (TPR)**: Outro nome para recall.
- **Taxa de Falsos Positivos (FPR)**: Proporção de falsos positivos em relação ao total de instâncias negativas.

## 7. Interpretação da AUC (Área sob a Curva)

A AUC mede a capacidade do modelo de classificar corretamente as instâncias. Um valor de AUC varia de 0 a 1:

- **AUC = 1**: Modelo perfeito.
- **AUC = 0,5**: Modelo não é melhor do que um classificador aleatório.
- **AUC < 0,5**: Modelo é pior que um classificador aleatório.

## Exemplos Práticos

- **Modelo A**: TP = 70, FP = 10, TN = 80, FN = 30.
- Precision = 0,88, Recall = 0,70, Especificidade = 0,89, AUC = 0,85.

- **Modelo B**: TP = 50, FP = 20, TN = 70, FN = 60.
- Precision = 0,71, Recall = 0,45, Especificidade = 0,78, AUC = 0,65.

## Conclusão

As métricas de avaliação são essenciais para entender a performance de modelos de classificação. A escolha das métricas corretas depende do problema específico e das consequências de diferentes tipos de erros.


# Problemas Binários em Machine Learning

## Interpolação em Problemas Binários
Em problemas de classificação binária, como a previsão de uma nota com base nas horas de estudo (por exemplo, se um aluno passou ou não), a interpolação tem algumas características importantes:

1. **Limitação entre 0 e 1**: A saída da função interpoladora deve estar restrita ao intervalo [0, 1], já que estamos lidando com probabilidades. Isso é crucial para interpretar os resultados como uma chance de sucesso (passar) ou fracasso (não passar).

2. **Monotonicidade**: A função interpoladora deve ser monótona, ou seja, se as horas de estudo aumentam, a probabilidade de passar não deve diminuir. Essa propriedade garante que, com mais estudo, a chance de sucesso aumente ou, no mínimo, permaneça constante.

3. **Continuidade**: A função deve ser contínua para evitar saltos bruscos nas previsões, o que poderia resultar em classificações inconsistentes.

4. **Derivabilidade**: A função deve ser derivável, o que permite otimização e aplicação de técnicas de gradiente, essenciais para o treinamento de modelos.

Dada essas condições, a **função sigmoide** é uma escolha popular para modelar problemas binários. Sua forma S permite mapear valores reais para o intervalo [0, 1], apresentando as características desejadas de monotonicidade e continuidade.

## Regressão Logística
A regressão logística é um modelo amplamente utilizado para problemas de classificação binária. Ao contrário da regressão linear, que prevê valores contínuos, a regressão logística utiliza a função sigmoide para prever a probabilidade de um evento (como passar ou não).

- **Função de Perda Entropia Cruzada**: A entropia cruzada é frequentemente usada como função de perda para a regressão logística. Ela mede a diferença entre a distribuição real das classes e a distribuição prevista pelo modelo. Minimizar a entropia cruzada ajuda a otimizar a previsão de probabilidades, tornando-as mais próximas das classes reais.

## Classificação Multiclasse
Embora a regressão logística seja ideal para problemas binários, é possível estendê-la para classificações multiclasse, onde se deseja prever mais de duas classes. Existem algumas abordagens para isso:

- **One-vs-Rest (OvR)**: Em vez de treinar um único modelo, a abordagem OvR envolve treinar um classificador binário para cada classe, considerando cada classe como um caso positivo e todas as outras como negativas.

- **Softmax**: Para problemas de múltiplas classes, a função Softmax é frequentemente utilizada em vez da sigmoide. Ela transforma as saídas do modelo em probabilidades que somam 1, permitindo interpretar as previsões como a probabilidade de pertencer a cada classe.

## Modelo de Regressão Logística Multiclasse
A **regressão logística multiclasse** combina a lógica da regressão logística com a abordagem Softmax, possibilitando prever a classe correta entre várias opções. O modelo calcula uma probabilidade para cada classe e escolhe a classe com a maior probabilidade como a previsão final.

## Conclusão
Em problemas de previsão binária e multiclasse, a escolha de funções adequadas e a compreensão de suas propriedades são fundamentais para o sucesso do modelo. A regressão logística e suas extensões oferecem uma base sólida para abordar esses problemas, garantindo que as previsões sejam coerentes e interpretáveis.


# Erro Relativo x Erro Absoluto
Utilizar um erro relativo (a razão entre o valor predito e o valor real) em vez do erro absoluto (a diferença entre os valores predito e real) pode ser vantajoso em várias situações, especialmente quando os valores reais variam em uma escala muito ampla. Aqui estão as principais razões para isso:

1. Escala Invariável
O erro relativo é independente da escala dos dados. Se você está trabalhando com variáveis cujos valores podem variar significativamente (por exemplo, preços de casas que vão de $100.000 a $10.000.000), um erro absoluto não é tão informativo. Uma diferença de $1.000 pode ser insignificante para casas caras, mas significativa para casas baratas.
O erro relativo normaliza essa diferença, tornando-o um valor proporcional ao valor real. Por exemplo, um erro relativo de 10% é interpretado da mesma forma, seja o valor real $1.000 ou $1.000.000.
2. Melhor Interpretação em Aplicações Práticas
Em muitas aplicações (como previsão de preços, vendas, ou outras métricas financeiras), o impacto de um erro depende do valor absoluto do número que você está prevendo. Se você está prevendo a receita de uma empresa, errar por $1.000 é muito mais aceitável quando a receita esperada é $1.000.000 (0,1% de erro) do que quando ela é $10.000 (10% de erro).
O erro relativo permite avaliar o desempenho do modelo de forma mais justa, levando em consideração a magnitude dos valores reais.
3. Reduz o Efeito de Outliers
Erros absolutos podem ser desproporcionalmente influenciados por outliers (valores extremos). Se um modelo prevê um valor muito alto ou muito baixo para um único ponto que é um outlier, isso pode distorcer a métrica de erro global.
Usar um erro relativo reduz esse impacto porque, mesmo para valores extremos, o erro será avaliado em relação ao próprio valor.
4. Mais Apropriado para Certos Tipos de Modelos
Em problemas onde as variáveis são multiplicativas ou seguem distribuições que são melhor modeladas em escala logarítmica, usar um erro baseado na razão é mais adequado. Por exemplo, modelos de crescimento exponencial ou de séries temporais muitas vezes se beneficiam mais do uso de um erro relativo.


# Explicação de Modelos de Machine Learning

## 1. Regressão Logística
A Regressão Logística é um modelo de classificação (apesar do nome) usado para prever uma variável binária (0 ou 1). Ele estima a probabilidade de uma amostra pertencer a uma classe usando a **função sigmoide**:

### Funcionamento:
- A função sigmoide mapeia os resultados para o intervalo entre 0 e 1:
  \[
  \text{sigmoide}(z) = \frac{1}{1 + e^{-z}}
  \]
- O modelo calcula uma pontuação \(z\) usando uma combinação linear das características:
  \[
  z = w_0 + w_1x_1 + w_2x_2 + \ldots + w_nx_n
  \]
- A função sigmoide transforma \(z\) em uma probabilidade, e com base em um limiar (0,5), o modelo decide a classe (0 ou 1).

### Vantagens:
- Fácil de interpretar.
- Eficiente para problemas lineares.

### Desvantagens:
- Não funciona bem para problemas não lineares.
- Sensível a outliers.

---

## 2. Support Vector Machines (SVM)
O SVM é um modelo de classificação que busca encontrar o **hiperplano** que melhor separa as classes no espaço de características.

### Funcionamento:
- O objetivo é encontrar o hiperplano que maximiza a margem entre as classes (distância máxima dos pontos mais próximos de cada classe, chamados de **vetores de suporte**).
- Para problemas não linearmente separáveis, o SVM usa **kernels** para transformar o espaço de características.

### Vantagens:
- Eficaz para alta dimensionalidade.
- Funciona bem para problemas não lineares com kernels.

### Desvantagens:
- Pode ser caro computacionalmente.
- Sensível à escolha do kernel e aos parâmetros.

---

## 3. Árvore de Decisão
Uma árvore de decisão divide os dados em grupos baseados nas características, formando uma estrutura hierárquica de decisões.

### Funcionamento:
- Cada nó representa uma característica, e cada ramo representa uma decisão baseada nessa característica.
- O objetivo é criar divisões que aumentem a "pureza" das classes, minimizando a impureza usando métricas como **Gini** ou **Entropia**.

### Vantagens:
- Fácil de interpretar e visualizar.
- Não precisa de normalização dos dados.

### Desvantagens:
- Propensa a overfitting.
- Instável para pequenas mudanças nos dados.

---

## 4. Modelos de Ensemble

Modelos de ensemble combinam múltiplos classificadores para melhorar a precisão e reduzir o risco de overfitting.

### Bagging (Bootstrap Aggregating):
- O **bagging** cria vários subconjuntos de dados de treino usando amostragem com reposição.
- Cada modelo é treinado nesses subconjuntos, e a decisão final é tomada pela **votação majoritária** (para classificação) ou média (para regressão).
- **Random Forest** é uma aplicação do bagging usando múltiplas árvores de decisão.

### Random Forest:
- Constrói múltiplas árvores de decisão, cada uma treinada em diferentes subconjuntos de dados e características.
- Reduz a variância e o risco de overfitting.

### Boosting:
- O **boosting** ajusta sequencialmente vários modelos fracos, onde cada modelo tenta corrigir os erros do anterior.
- Exemplos:
  - **AdaBoost**: Aumenta o peso dos exemplos mal classificados.
  - **Gradient Boosting**: Otimiza os erros residuais.

### Vantagens dos Modelos de Ensemble:
- Melhoram a precisão.
- Menos propensos a overfitting comparados a um único modelo.

### Desvantagens:
- Podem ser difíceis de interpretar.
- Custosos computacionalmente.

---

## 5. Introdução a Redes Neurais
Redes neurais são modelos inspirados no funcionamento do cérebro, compostas de **neurônios artificiais** organizados em camadas.

### Funcionamento:
- Cada neurônio recebe uma entrada, aplica uma **função de ativação** (como ReLU, Sigmoid), e passa o resultado para a próxima camada.
- A rede ajusta os pesos durante o treinamento para minimizar a função de perda usando o **backpropagation**.

### Arquitetura:
- **Rede Neural Simples**: Camada de entrada, uma ou mais camadas ocultas e uma camada de saída.
- **Deep Learning**: Redes com várias camadas ocultas para problemas complexos como reconhecimento de imagem.

### Vantagens:
- Detecta padrões complexos e dados não estruturados (imagens, texto).
- Aprende representações de alto nível automaticamente.

### Desvantagens:
- Requer grandes quantidades de dados e poder computacional.
- Difícil de interpretar (caixa preta).

---

## Resumo

| Modelo                  | Tipo          | Vantagem Principal             | Limitação Principal             |
|-------------------------|---------------|--------------------------------|---------------------------------|
| Regressão Logística     | Classificação | Interpretação simples          | Não funciona bem para dados não lineares |
| SVM                     | Classificação | Eficaz para alta dimensionalidade | Custo computacional alto        |
| Árvore de Decisão       | Classificação | Fácil de interpretar           | Propensa a overfitting          |
| Random Forest           | Ensemble      | Reduz overfitting              | Difícil de interpretar          |
| Boosting                | Ensemble      | Alta precisão                  | Sensível a outliers             |
| Redes Neurais           | Deep Learning | Alta capacidade de aprendizado | Requer muitos dados e poder computacional |
