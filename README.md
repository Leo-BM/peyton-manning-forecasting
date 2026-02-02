# 🏈 Análise e Previsão de Séries Temporais: Peyton Manning (Wikipedia Views)

![Status](https://img.shields.io/badge/Status-Concluído-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Lib](https://img.shields.io/badge/Lib-Prophet%20|%20NeuralProphet%20|%20PMDarima-orange)

Este projeto apresenta um estudo ponta a ponta de **Séries Temporais (Time Series Forecasting)**, focado na previsão de acessos diários à página da Wikipedia do ex-jogador da NFL, **Peyton Manning**.

O estudo compara abordagens estatísticas clássicas (**SARIMAX**) contra abordagens modernas baseadas em componentes aditivos (**Prophet**) e Redes Neurais (**NeuralProphet**), abordando desafios como sazonalidade complexa e eventos irregulares (Super Bowls).

---

## 📋 Índice
1. [Visão Geral e Objetivos](#-visão-geral-e-objetivos)
2. [O Dataset](#-o-dataset)
3. [Metodologia Aplicada](#-metodologia-aplicada)
4. [Análise Exploratória (EDA)](#-análise-exploratória-eda)
5. [Modelagem e Estratégia](#-modelagem-e-estratégia)
6. [Resultados: A Batalha dos Modelos](#-resultados-a-batalha-dos-modelos)
7. [Conclusões Técnicas](#-conclusões-técnicas)
8. [Como Executar](#-como-executar)

---

## 🎯 Visão Geral e Objetivos

O comportamento de busca na internet por personalidades esportivas não segue um padrão linear. Ele é influenciado geralmente por Sazonalidade do esporte em questão e para eventos de impacto. Em nosso caso de estudo, para um jogador influente como foi Peyton Manning na NFL (National Football League), esses fatores se refletem da seguinte maneira: 
* **Sazonalidade Semanal:** Jogos aos domingos/segundas são os mais visualizados.
* **Sazonalidade Anual:** Temporada da NFL (Setembro a Fevereiro) vs. Off-season (período de férias).
* **Eventos de Impacto (Outliers):** Super Bowls (finais de campeonato), trocas de time e aposentadoria.

**Objetivos do Projeto:**
1.  Decompor a série temporal para entender tendências e ciclos.
2.  Validar a estacionariedade da série estatisticamente.
3.  Implementar um modelo **Prophet** robusto a feriados e eventos especiais.
4.  Realizar **Backtesting (Cross-Validation)** com janela deslizante.
5.  Comparar a performance preditiva (Horizonte de 365 dias) entre **Prophet**, **SARIMAX** e **NeuralProphet**.

---

## 💾 O Dataset

O dataset contém registros diários de visualizações (views) da página "Peyton Manning" na Wikipedia.
* **Período:** Dez/2007 a Jan/2016.
* **Transformação Logarítmica:** A variável alvo `y` já se encontra transformada em Log Natural (`log(views)`).
    * *Justificativa:* A série original possui relação **multiplicativa** (a variância cresce conforme a média aumenta). A aplicação do Log estabiliza a variância, permitindo o uso de modelos **aditivos** e reduzindo o impacto de outliers extremos.

---

## ⚙️ Metodologia Aplicada

### 1. Pré-Processamento e EDA
* Conversão temporal e análise de frequência diária.
* **Boxplots Sazonais:** Identificação visual clara de aumento de volatilidade e média durante a temporada da NFL (Set-Fev) e picos às Segundas-feiras.
* **Decomposição Sazonal:** Uso do `seasonal_decompose` (Statsmodels) para isolar Tendência, Sazonalidade e Resíduos.

### 2. Validação Estatística
* **Teste Augmented Dickey-Fuller (ADF):** * *Resultado:* P-valor < 0.05.
    * *Conclusão:* A série (em log) é **Estacionária**. Não foi necessário aplicar diferenciação ($d=0$ no ARIMA).
* **Autocorrelação (ACF/PACF):** Confirmação de forte correlação serial e sazonalidade de curto prazo.

### 3. Modelagem com Prophet
* Configuração de modelo **Aditivo** (devido ao Log).
* **Feature Engineering de Feriados:** Mapeamento manual de eventos críticos (Super Bowl XLIV, XLVIII, 50). Isso permitiu ao modelo antecipar picos que métodos tradicionais tratariam como erro aleatório.

### 4. Avaliação de Performance (Backtesting)
Utilizou-se a técnica de **Cross-Validation com Janela em Expansão (Expanding Window)**:
* **Initial:** 730 dias de treino.
* **Period:** Testes a cada 180 dias.
* **Horizon:** Previsão de 365 dias à frente.

---

## 📊 Resultados: A Batalha dos Modelos

Para a avaliação final, separamos os últimos **365 dias** do dataset como conjunto de **Teste (Oculto)** e treinamos os modelos com o restante.

| Modelo | RMSE (Log Scale) | MAPE (Log Scale) | Características Observadas |
| :--- | :---: | :---: | :--- |
| **Prophet** | **0.35** | Excelente estabilidade em longo prazo. Capturou os picos de playoffs/Super Bowl devido ao mapeamento de feriados. |
| **NeuralProphet** | 0.38 | Muito próximo do Prophet. Demonstrou capacidade de generalização, mas sofreu levemente com overfitting nos ruídos diários. |
| **SARIMAX** | 1.5983 | **Pior desempenho.** Em horizontes longos (365 dias), o modelo convergiu para a média (Mean Reversion), falhando em prever a dinâmica da temporada seguinte. |

> **Nota Crítica sobre o MAPE:** O erro percentual médio de ~5% refere-se à escala logarítmica. Na escala real de visualizações, a variação absoluta é maior, dada a natureza exponencial da transformação inversa.

---

## 🧠 Conclusões Técnicas

1.  **A Importância do "Human-in-the-Loop":** O **Prophet** venceu não apenas pela matemática, mas porque permitiu a injeção de conhecimento de domínio (datas dos Super Bowls). O modelo sabia *quando* o pico ocorreria, enquanto os outros tentaram inferir apenas pelo padrão passado.
2.  **Limitações do ARIMA/SARIMAX:** Para previsões de **curto prazo** (ex: 7 dias), o SARIMAX é excelente. Porém, para um horizonte de **1 ano**, ele perde a "memória" dos eventos exógenos e tende a fornecer uma previsão conservadora (flat line), o que é inútil para planejamento de longo prazo.
3.  **NeuralProphet:** Mostrou-se uma alternativa promissora. Com mais dados e ajuste fino de hiperparâmetros (learning rate, epochs), poderia superar o Prophet clássico em capturar não-linearidades sutis.

---

## 🚀 Como Executar

Certifique-se de ter Python 3.8+ instalado.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Leo-BM/peyton-manning-forecasting.git](https://github.com/Leo-BM/peyton-manning-forecasting.git)
    cd peyton-manning-timeseries
    ```

2.  **Instale as dependências:**
    ```bash
    pip install pandas numpy matplotlib seaborn plotly statsmodels pmdarima prophet neuralprophet
    ```
    *(Nota: NeuralProphet pode exigir downgrade do numpy dependendo da versão. Use `numpy<2.0` se necessário ou aplique o patch contido no notebook).*

3.  **Execute o Notebook:**
    Abra o arquivo `peyton_manning_wiki_time_series.ipynb` no Jupyter ou Google Colab.
4.  **OBS: Não esqueça de baixar a base de dados que encontra-se neste reposítório e carregar em seu notebook**
---

### 📬 Contato
Desenvolvido por **[Leonardo Bento Maria]** *Desenvolvedor de Software Full Stack e Estudante de Data Science & Machine Learning* [LinkedIn](https://www.linkedin.com/in/leonardo-bento-maria) 
