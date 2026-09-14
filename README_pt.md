# ProjetoGrassQuality — Visualizador de Qualidade de Pastagem

Um protótipo que estima a qualidade de pastagem — **Baixa / Média / Alta** — para qualquer
ponto do mapa no Brasil, combinando os **embeddings de satélite AlphaEarth** do Google Earth
Engine com um classificador pequeno treinado sobre rótulos do **MapBiomas Pastagem
(Qualidade)**.

Dado um par lat/lon, o app busca a sequência de embeddings AlphaEarth de múltiplos anos
daquele ponto no Earth Engine, passa essa sequência por um classificador temporal treinado e
mostra a classe prevista junto com as probabilidades de cada classe.

## Como funciona

1. **Representação base — AlphaEarth (Google Earth Engine)**
   Em vez de construir índices espectrais manualmente (NDVI, etc.), o projeto usa a coleção
   `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` do Earth Engine — os embeddings de satélite
   pré-treinados do AlphaEarth. Cada pixel/ano já vem reduzido pelo modelo fundacional do
   Google a um vetor de 64 dimensões (bandas `A00`…`A63`) que resume o sinal de satélite
   daquele local no ano (óptico, radar, textura, etc.), sem necessidade de baixar ou processar
   imagens brutas manualmente.

2. **Rótulos — MapBiomas Pastagem**
   Os pontos de treino são rotulados usando a classificação de qualidade de pastagem do
   MapBiomas, agrupada em 3 classes: **Baixa, Média, Alta** qualidade. O mapeamento de rótulos
   guardado junto aos dados de treino (`model/grassland_embeddings_dataset.npz`) converte os
   códigos originais de qualidade do MapBiomas (1, 2, 3) para os índices de classe do modelo
   (0, 1, 2).

3. **Dataset**
   Para cada ponto de amostra rotulado, o embedding AlphaEarth foi extraído para uma sequência
   de anos (2018–2023, 6 anos) e empilhado, de forma que um exemplo de treino é
   `(6 anos × embedding de 64 dimensões) → classe de qualidade`. O array resultante está
   versionado em `model/grassland_embeddings_dataset.npz` (`X`: `(300, 6, 64)` float32, `y`:
   `(300,)` int64, balanceado em 100/100/100 entre as 3 classes; `years`:
   `[2018 2019 2020 2021 2022 2023]`; `label_map`: código MapBiomas → índice de classe).

4. **Modelo — um classificador temporal pequeno sobre os embeddings**
   Em vez de classificar um único ano isoladamente, um `GrasslandTemporalClassifier` leve
   (definido em [app.py](app.py)) consome toda a sequência de embeddings de 6 anos de um ponto:
   - Uma **LSTM** de 1 camada (`input_dim=64`, `hidden_dim=64`) sobre a sequência de embeddings
     anuais
   - Uma **MLP** pequena como cabeça (`Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→3)`)
     aplicada ao estado oculto final da LSTM, gerando os logits sobre as 3 classes de qualidade
   - Softmax na inferência transforma os logits em probabilidades de Baixa/Média/Alta

   Esse modelo foi treinado (fora deste snapshot do repositório) no dataset acima, e o
   checkpoint resultante está versionado em `model/grassland_temporal_classifier.pt`.

5. **App — interface de mapa em Streamlit ([app.py](app.py))**
   Um app de arquivo único em Streamlit que:
   - Permite clicar em um ponto no mapa de satélite/ruas (via `folium` / `streamlit-folium`)
   - Busca o embedding AlphaEarth desse ponto para cada ano do intervalo configurado
     diretamente no Earth Engine (`ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")`)
   - Carrega um checkpoint de `GrasslandTemporalClassifier` que você envia pelo upload
     (os parâmetros de arquitetura — número de classes, tamanho oculto, número de camadas —
     são configuráveis na barra lateral para poderem ser ajustados a qualquer checkpoint
     carregado, incluindo `model/grassland_temporal_classifier.pt`)
   - Passa a sequência de embeddings pelo modelo e mostra a classe prevista e um gráfico de
     barras com as probabilidades, além dos vetores de embedding brutos por ano

## Estrutura do projeto

```
ProjetoGrassQuality/
├── app.py                                  # App Streamlit: interface de mapa + busca no EE + inferência
├── requirements.txt                        # Dependências Python
├── model/
│   ├── grassland_embeddings_dataset.npz     # Dados de treino: embeddings AlphaEarth + rótulos MapBiomas
│   └── grassland_temporal_classifier.pt     # Checkpoint do classificador LSTM treinado
└── README.md / README_pt.md
```

## Rodando localmente

### Pré-requisitos

- Python 3.10+
- Um projeto no Google Cloud com a **API do Earth Engine** habilitada, e acesso ao Earth
  Engine (cadastre-se em https://earthengine.google.com caso ainda não tenha)

### Configuração

```bash
# a partir da raiz do projeto
python -m venv venv
venv\Scripts\activate        # Windows (PowerShell: venv\Scripts\Activate.ps1)
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Autenticar o Earth Engine (uma vez)

Execute isso uma vez em um terminal no mesmo ambiente — isso armazena as credenciais em
disco. O app em si não executa o fluxo interativo de OAuth:

```bash
earthengine authenticate
```

### Executar o app

```bash
streamlit run app.py
```

Isso abre o app no navegador. Em seguida:

1. Informe seu ID de projeto GCP do Earth Engine na barra lateral.
2. Envie um checkpoint de modelo — use `model/grassland_temporal_classifier.pt` para o modelo
   treinado incluído neste repositório.
3. Garanta que as configurações de arquitetura na barra lateral correspondem ao checkpoint:
   `num_classes=3`, `hidden_dim=64`, `num_layers=1`, anos de embedding `2018–2023` (esses são
   os valores padrão, e correspondem a `model/grassland_temporal_classifier.pt`).
4. Clique em um ponto no mapa e depois em **"Fetch embeddings and predict"** para ver a
   qualidade de pastagem prevista (Baixa / Média / Alta) e as probabilidades de cada classe
   para aquele local.
