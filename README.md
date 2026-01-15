# 📊 Customer Segmentation com RFM e Clustering Não Supervisionado

## 📌 Introdução

A segmentação de clientes é uma das estratégias mais importantes para empresas de e-commerce, pois permite compreender diferentes perfis de consumo, otimizar campanhas de marketing, melhorar retenção e aumentar o valor do cliente ao longo do tempo.

Neste projeto, foi aplicada a metodologia **RFM (Recency, Frequency, Monetary)** combinada com técnicas de **clusterização não supervisionada** para segmentar clientes de um e-commerce real, utilizando o dataset **Online Retail**.

Diferente de modelos supervisionados, **não há rótulos pré-definidos** indicando o perfil de cada cliente. O objetivo é descobrir padrões ocultos nos dados e agrupar clientes com comportamentos semelhantes de forma automática, apoiando decisões estratégicas de negócio.

---

## ❓ O que é Clustering Não Supervisionado?

A **clusterização não supervisionada** é uma técnica de aprendizado de máquina onde algoritmos identificam grupos naturais nos dados sem conhecimento prévio das classes.

Neste projeto:
- Não sabemos previamente quem são clientes VIP, regulares ou inativos
- Os grupos são formados com base no comportamento de compra
- A interpretação final dos clusters é orientada por métricas de negócio

---

## 🎯 Objetivos do Projeto

- Segmentar clientes com base no comportamento de compra
- Identificar clientes VIP, regulares e inativos
- Apoiar estratégias de retenção, reativação e fidelização
- Aplicar RFM combinado com PCA e KMeans
- Gerar insights acionáveis para o negócio

---

## 📁 Estrutura do Projeto
```text
notebooks/
├── 01_Data_cleaning.ipynb
├── 02_RFM_Analysis.ipynb
├── 03_modeling_pca_kmeans.ipynb
└── 04_Clustering.ipynb
data/
├── raw.csv
├── cleaned.csv
├── rfm.csv
├── rfm_scaled.csv
├── rfm_pca.csv
└── rfm_modeled.csv
reports/
└── clustering_report.pdf
README.md
requirements.txt
```
---

## 🔍 Etapas do Projeto

### 1️⃣ Data Cleaning

Nesta etapa foi realizada a preparação inicial dos dados, garantindo qualidade e consistência para as análises posteriores.

Principais ações:
- Remoção de registros inválidos (quantidade e preço ≤ 0)
- Tratamento de pedidos cancelados
- Criação da variável **Revenue**
- Análise exploratória dos clientes que cancelaram compras
- Geração do dataset limpo para análises posteriores

**Ferramentas:** Pandas, NumPy, Matplotlib

---

### 2️⃣ RFM Analysis

Foi aplicada a metodologia RFM para entender o comportamento de compra dos clientes.

Atividades realizadas:
- Cálculo das métricas:
  - **Recency:** dias desde a última compra
  - **Frequency:** número de compras
  - **Monetary:** valor total gasto
- Análises estatísticas e visuais com e sem outliers
- Segmentação temporal de recency
- Análises orientadas a negócio:
  - Top produtos
  - Países mais relevantes
  - Produtos mais lucrativos
  - Tendências de compra

📌 Os outliers foram preservados na análise de negócio por representarem clientes reais e estratégicos.

**Ferramentas:** Pandas, Matplotlib, Seaborn

---

### 3️⃣ Modeling (PCA & KMeans)

Para garantir estabilidade matemática e melhor separação dos clusters, foi realizado o pré-processamento dos dados antes da modelagem.

Procedimentos:
- Tratamento de outliers via **Winsorize** (aplicado apenas para modelagem)
- Padronização dos dados com **StandardScaler**
- Redução de dimensionalidade com **PCA**
- Visualização dos dados em 2D e 3D
- Aplicação do algoritmo **KMeans** para clusterização

**Ferramentas:** Scikit-learn

---

### 4️⃣ Clustering e Interpretação

Os clusters obtidos foram analisados e interpretados com base em métricas de negócio, resultando nos seguintes segmentos:

- **Clientes VIP (~8%)**  
  Clientes com menor recency, alta frequência e alto gasto. Devem ser fidelizados com benefícios exclusivos, acesso antecipado a novidades e programas de indicação.

- **Clientes Regulares (~67%)**  
  Apresentam bom comportamento de compra, mas com potencial de crescimento. Estratégias de incentivo à recompra e campanhas progressivas podem migrá-los para o grupo VIP.

- **Clientes Inativos (~25%)**  
  Clientes com longo tempo sem compras. Devem receber campanhas de reativação, cupons com prazo curto, comunicação personalizada e ofertas de baixo custo.

---

## 🚀 Conclusão

O projeto demonstrou como técnicas de segmentação baseadas em RFM e clustering não supervisionado podem gerar insights valiosos para o negócio. A abordagem permitiu identificar diferentes perfis de clientes e propor estratégias específicas de retenção, fidelização e reativação, apoiando decisões orientadas por dados.
Com as informações obtidas podemos pensar em soluções de retenção para cada grupo e fazer campanhas de marketing segmentadas.

---

## 🛠 Tecnologias Utilizadas

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
  - **StandardScaler**
  - **PCA**
  - **KMeans**
