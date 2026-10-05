Projeto de Deep Learning — Detecção de Atividades Anômalas na Amazônia
A proposta do projeto será desenvolver um sistema de detecção não supervisionada de anomalias ambientais em regiões amazônicas utilizando imagens de satélite e Deep Learning.

A ideia não é treinar diretamente um classificador para reconhecer "garimpo", "desmatamento" ou "queimada". O objetivo é mais geral: aprender como uma região preservada normalmente se apresenta e evolui ao longo do tempo e detectar alterações anômalas, como surgimento de clareiras, perda de vegetação, queimadas, mineração, abertura de estradas ou alterações em corpos d'água.

Temos também um alvo real: se os resultados forem bons, a ideia é transformar o projeto em um artigo científico e submetê-lo ao:

IEEE IGARSS 2027 — International Geoscience and Remote Sensing Symposium

Local: Reykjavík, Islândia
Data: 11 a 16 de julho de 2027
Local do evento: Harpa Concert Hall & Conference Centre

O IGARSS 2027 possui temas diretamente relacionados ao projeto, como Foundation Models and Embeddings for Earth Observation, Change Detection and Temporal Analysis, Multimodal Data Fusion e Forest and Vegetation.

Dados
Vamos trabalhar inicialmente com imagens de satélite Sentinel organizadas espacial e temporalmente.

A ideia é observar a mesma região em diferentes datas:

X(t1), X(t2), X(t3), ..., X(tn)

e detectar mudanças incompatíveis com seu comportamento histórico. 
Modelo 1 — Engenharia de Features
O primeiro modelo será um Foundation Model específico para Earth Observation:

TerraMind 1.0 — IBM + ESA

Inicialmente utilizaremos o TerraMind Small.

O objetivo desse modelo será transformar cada imagem ou patch em uma representação profunda:

Imagem de satélite
        |
        v
TerraMind Small
        |
        v
Embedding geoespacial

Em notação simples:

X(t) -> TerraMind -> Z(t)

Ou seja, em vez de criarmos manualmente todas as features, utilizaremos um modelo moderno de Earth Observation para extrair representações relevantes da superfície terrestre.
Modelo 2 — Detecção de Anomalias
O segundo modelo será responsável por aprender o comportamento normal da região ao longo do tempo.

Modelo principal:

Temporal Transformer Autoencoder

Ele receberá uma sequência de embeddings:

Z(t-k), ..., Z(t-2), Z(t-1), Z(t)

e deverá aprender a reconstruir o comportamento temporal esperado.

De forma simplificada:

sequência real -> Transformer Autoencoder -> sequência reconstruída

O erro entre o comportamento observado e o comportamento reconstruído será utilizado como Anomaly Score:

Anomaly Score = distância(Z real, Z reconstruído)

Quanto maior essa distância, mais anômala é aquela região naquele instante.

A saída final deverá permitir produzir um mapa espacial indicando regiões com maior probabilidade de alteração anormal.
Pipeline
Imagens de satélite
        |
        v
Pré-processamento
        |
        v
Divisão em patches
        |
        v
Construção das séries temporais
        |
        v
TerraMind Small
        |
        v
Embeddings geoespaciais
        |
        v
Sequências temporais de embeddings
        |
        v
Temporal Transformer Autoencoder
        |
        v
Anomaly Score
        |
        v
Mapa de regiões anômalas
        |
        v
Comparação com eventos reais conhecidos

Também teremos alguns baselines para comparação:

Features espectrais -> Isolation Forest

TerraMind -> Isolation Forest

TerraMind -> Deep SVDD

Assim conseguimos responder uma questão importante:

Um Foundation Model de Earth Observation melhora a detecção não supervisionada de perturbações ambientais na Amazônia?
Parte principal da pesquisa
O modelo não deverá precisar conhecer previamente todas as anomalias.

A ideia é treiná-lo majoritariamente com regiões e períodos considerados normais e depois verificar se consegue identificar perturbações como:

desmatamento;

queimadas;

mineração ou garimpo;

abertura de estradas;

alterações em rios e corpos d'água;

outras mudanças incomuns na superfície.

Em outras palavras:

aprender o comportamento normal
                |
                v
       detectar o desconhecido

A entrega no final de novembro será:

pipeline funcionando;

modelos implementados e treinados;

experimentos executados;

métricas;

mapas de anomalias;

código reproduzível;

resultados salvos para análise posterior.

Se os resultados forem cientificamente interessantes, durante as férias vamos organizar os experimentos e transformar o projeto em um artigo para tentar submissão ao IGARSS 2027, em Reykjavík, Islândia.

A ideia é desenvolver isso desde o início como um projeto de pesquisa, e não apenas como um trabalho para nota.