# ProjetoGrassQuality — Visualizador de Qualidade de Pastagem

Um protótipo que estima a qualidade de pastagem — **Baixa / Média / Alta** — para qualquer
ponto do mapa no Brasil, combinando os **embeddings de satélite AlphaEarth** do Google Earth
Engine com um classificador pequeno treinado sobre rótulos do **MapBiomas Pastagem
(Qualidade)**.

Dado um par lat/lon, o app busca a sequência de embeddings AlphaEarth de múltiplos anos
daquele ponto no Earth Engine, passa essa sequência por um classificador temporal treinado e
mostra a classe prevista junto com as probabilidades de cada classe.

## Duas frentes neste repositório

Este repositório cobre duas coisas:

1. **`notebooks/` — o pipeline mínimo exigido pela AC** (IBM8924 — AC de Projeto 2026.2,
   Grupo 1): recortes brutos de imagem Sentinel-2 (não embeddings) sobre os mesmos pontos
   de vigor de pastagem do MapBiomas, um baseline trivial e um baseline raso de regressão
   logística — ver [Pipeline mínimo exigido](#pipeline-mínimo-exigido-notebooks) abaixo.
2. **`model/` + `app.py` — um modelo adicional mais avançado** (embeddings AlphaEarth + um
   classificador LSTM, mais uma interface de mapa em Streamlit), construído antes do
   pipeline de imagem bruta acima e mantido como um bônus além do mínimo da AC. Descrito a
   seguir.

## Modelo avançado adicional (bônus, além do mínimo da AC): embeddings AlphaEarth + LSTM

Não é exigido pela rubrica da AC (que pede um tensor de imagem bruta e um baseline raso —
ver [Pipeline mínimo exigido](#pipeline-mínimo-exigido-notebooks)), mas é mantido por ser
uma abordagem alternativa funcional e já validada, que vale a pena documentar.

1. **Representação base — AlphaEarth (Google Earth Engine)**
   Em vez de construir índices espectrais manualmente (NDVI, etc.), o projeto usa a coleção
   `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` do Earth Engine — os embeddings de satélite
   pré-treinados do AlphaEarth. Cada pixel/ano já vem reduzido pelo modelo fundacional do
   Google a um vetor de 64 dimensões (bandas `A00`…`A63`) que resume o sinal de satélite
   daquele local no ano (óptico, radar, textura, etc.), sem necessidade de baixar ou processar
   imagens brutas manualmente.

2. **Rótulos — MapBiomas Pastagem (Vigor)**
   Os pontos de treino são rotulados usando o ativo de **Vigor de Pastagem** do MapBiomas
   (`mapbiomas_collection90_pasture_vigor_v1`, ano de referência 2022), usado como
   substituto para "qualidade" — **1/2/3 = vigor baixo/médio/alto**. A camada de vigor é
   mascarada para conter apenas pixels de pastagem usando o ativo de uso e cobertura do solo
   do MapBiomas (código de classe `15` = Pastagem) antes da amostragem, de forma que os
   rótulos vêm apenas de áreas que o próprio MapBiomas classifica como pastagem. Como o mapa
   de vigor do MapBiomas é, ele mesmo, saída de um modelo (e não verdade de campo), é melhor
   tratá-lo como um sinal de treino grande, porém um tanto ruidoso — o notebook aponta o uso
   de pontos validados em campo pelo LAPIG como próximo passo natural.

   Os pontos rotulados são extraídos com `ee.Image.stratifiedSample` (100 pontos por classe
   de vigor, escala de 30 m, seed 42), garantindo que as 3 classes já saiam balanceadas,
   sobre uma região de teste de ~65 km × 65 km (~4.977 km²) ao redor de Goiânia, Goiás —
   escolhida por existirem dados publicados e validados em campo sobre degradação de
   pastagem naquela região, úteis para comparações futuras.

3. **Dataset**
   Para cada um dos 300 pontos de amostra rotulados, o embedding AlphaEarth foi extraído em
   cada ponto (`sampleRegions`, escala de 10 m) para uma sequência de anos (2018–2023, 6
   anos) e empilhado, de forma que um exemplo de treino é `(6 anos × embedding de 64
   dimensões) → classe de vigor`. Apenas pontos com sequência completa de 6 anos são
   mantidos (300/300 nesta execução). O array resultante está versionado em
   `model/grassland_embeddings_dataset.npz` (`X`: `(300, 6, 64)` float32, `y`: `(300,)`
   int64, balanceado em 100/100/100 entre as 3 classes; `years`:
   `[2018 2019 2020 2021 2022 2023]`; `label_map`: código de vigor do MapBiomas (1/2/3) →
   classe do modelo indexada a partir de zero (0/1/2)).

4. **Modelo — um classificador temporal pequeno sobre os embeddings**
   Em vez de classificar um único ano isoladamente, um `GrasslandTemporalClassifier` leve
   (definido em [app.py](app.py)) consome toda a sequência de embeddings de 6 anos de um ponto:
   - Uma **LSTM** de 1 camada (`input_dim=64`, `hidden_dim=64`) sobre a sequência de embeddings
     anuais
   - Uma **MLP** pequena como cabeça (`Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→3)`)
     aplicada ao estado oculto final da LSTM, gerando os logits sobre as 3 classes de qualidade
   - Softmax na inferência transforma os logits em probabilidades de Baixa/Média/Alta

   Esse modelo é treinado em `model/grassland_quality_finetuning.ipynb` — a etapa de
   fine-tuning, já que o AlphaEarth já fez o pré-treinamento auto-supervisionado caro, e
   este notebook treina apenas a cabeça leve e específica da tarefa em cima dele. Detalhes
   do treinamento:
   - Divisão treino/validação/teste 70/15/15 (210/45/45 dos 300 pontos), batch size 16,
     seed 42
   - Otimizador Adam, `lr=1e-3`, `weight_decay=1e-4`, cross-entropy loss, 30 épocas
   - Atingiu ~80% de acurácia de validação até a época 30 (`val_loss 0.693`,
     `val_acc 0.800`), avaliado no conjunto de teste separado com um classification report
     e matriz de confusão

   O checkpoint resultante está versionado em `model/grassland_temporal_classifier.pt`. O
   notebook também inclui um fallback opcional com dados sintéticos (`USE_SYNTHETIC = True`)
   para testar o pipeline em PyTorch sem uma conexão funcional com o Earth Engine.

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

## Limitações conhecidas / próximos passos

Este é um primeiro scaffold funcional, não um pipeline finalizado (veja também as notas
finais do próprio notebook):

- Os rótulos vêm do mapa de **Vigor de Pastagem** do MapBiomas (ele mesmo saída de um
  modelo), não de verdade de campo — usar ou combinar pontos validados em campo pelo LAPIG
  daria um conjunto de avaliação mais robusto.
- As previsões são por ponto, não por propriedade — um próximo passo natural é agregar
  embeddings sobre os polígonos de propriedades do CAR (pooling por média/percentil) assim
  que houver limites reais de propriedade disponíveis.
- "Qualidade" aqui significa apenas vigor da vegetação; ainda não incorpora sinais de risco
  legal/ambiental (limites do CAR, alertas de desmatamento PRODES/DETER, status de embargo).
- Vale reconfirmar o código de classe de Pastagem do MapBiomas (`15`) e os nomes de
  ativos/bandas junto à legenda atual em brasil.mapbiomas.org, já que podem mudar entre
  versões das coleções do MapBiomas.

## Pipeline mínimo exigido (`notebooks/`)

O enunciado da AC (`TODO.md`, Grupo 1) exige um dataset de **imagens** rotulado (um tensor
bruto `X ∈ R^(N×H×W×C)`, não embeddings), ≥1500 recortes, um baseline trivial, um baseline
principal raso, e artefatos específicos de avaliação/caracterização. Como as fontes de
imagem bruta sugeridas no enunciado não cobrem o Brasil (EuroSAT = só Europa, UC Merced =
só EUA), este pipeline usa imagens brutas do **Sentinel-2 L2A** via Earth Engine —
reaproveitando os mesmos pontos de vigor de pastagem do MapBiomas e o mesmo projeto Earth
Engine do modelo bônus acima.

- **`notebooks/01_dados.ipynb`** — amostra pontos de vigor de pastagem em 4 anos
  (2019–2022, mesma AOI perto de Goiânia) para chegar a ≥1500 pontos balanceados, extrai
  um recorte Sentinel-2 de 64×64 (bandas B2/B3/B4/B8/B11/B12) por ponto, descarta recortes
  com excesso de pixels sem dado (nuvem), e divide 70/15/15 em treino/validação/teste por
  bloco espacial de grade (garantindo que recortes próximos nunca caiam em partições
  diferentes). Grava `data/processed/s2_patches.npz` e `data/processed/points_metadata.csv`.
- **`notebooks/02_caracterizacao.ipynb`** — distribuição de classes, estatísticas por
  banda, histogramas de NDVI/GNDVI, grade de exemplos por classe, e comentário sobre
  desbalanceamento/artefatos.
- **`notebooks/03_baseline.ipynb`** — baseline trivial (classe majoritária) vs. regressão
  logística sobre atributos agregados por recorte (médias/desvios por banda + NDVI/GNDVI);
  acurácia, F1 macro, matriz de confusão, e análise comentada de erros (≥4 exemplos).
- **`notebooks/dataset_utils.py`** — funções compartilhadas (`load_dataset`, NDVI/GNDVI,
  composição RGB, extração de atributos agregados) usadas pelos notebooks 02 e 03.

Rode em ordem: `01_dados.ipynb` → `02_caracterizacao.ipynb` → `03_baseline.ipynb`.
`01_dados.ipynb` precisa do mesmo acesso ao Earth Engine que o `app.py` e é a etapa mais
lenta (a extração de recortes roda em pequenos lotes síncronos contra o Earth Engine) —
*preencher o tempo real de execução aqui depois de rodar*.

## Estrutura do projeto

```
ProjetoGrassQuality/
├── app.py                                    # App Streamlit: interface de mapa + busca no EE + inferência (modelo bônus)
├── requirements.txt                          # Dependências Python
├── data/
│   └── processed/                            # E1: s2_patches.npz + points_metadata.csv (gerado por 01_dados.ipynb)
├── notebooks/                                # Pipeline mínimo exigido pela AC (E1-E4)
│   ├── dataset_utils.py                      # Funções compartilhadas: NDVI/GNDVI, RGB, atributos agregados
│   ├── 01_dados.ipynb                        # Ingestão: pontos MapBiomas + recortes Sentinel-2 + split
│   ├── 02_caracterizacao.ipynb               # Caracterização: estatísticas, histogramas de NDVI, grades de exemplo
│   └── 03_baseline.ipynb                     # Baseline trivial + regressão logística, métricas, análise de erros
├── model/                                    # Modelo avançado adicional (não exigido pelo mínimo da AC)
│   ├── grassland_quality_finetuning.ipynb    # Notebook de treino: extração no EE + fine-tuning da LSTM
│   ├── grassland_embeddings_dataset.npz      # Dados de treino: embeddings AlphaEarth + rótulos MapBiomas
│   └── grassland_temporal_classifier.pt      # Checkpoint do classificador LSTM treinado
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

### Reproduzir o pipeline mínimo exigido (notebooks/)

A partir de `notebooks/`, rode em ordem (cada um precisa do mesmo acesso ao Earth Engine
configurado acima):

1. `01_dados.ipynb` — *(preencher tempo real, ex. "~40 min para ~1800 pontos")*
2. `02_caracterizacao.ipynb` — *(preencher tempo real)*
3. `03_baseline.ipynb` — *(preencher tempo real)*

Versões de bibliotecas: ver `requirements.txt` (nenhuma instalação extra além dele —
`dataset_utils.py` é um módulo local importado pelos notebooks 02 e 03, não um pacote).

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

### Retreinando / reexecutando o notebook de fine-tuning (opcional)

O `model/grassland_quality_finetuning.ipynb` roda de forma independente (no Colab ou
localmente com Jupyter) e precisa do mesmo acesso ao Earth Engine que o app. Ele
reextrai os rótulos do MapBiomas e os embeddings do AlphaEarth, retreina o classificador
LSTM e grava `grassland_embeddings_dataset.npz` e `grassland_temporal_classifier.pt`.
Também tem um toggle `USE_SYNTHETIC = True` para testar o loop de treino sem acesso ao
Earth Engine.

## Divisão de tarefas

*(preencher antes da entrega — exigido pelo enunciado)*

| Integrante | Responsabilidades |
|---|---|
| *Gabriel Faleiro Lam Pow* | *`01_dados.ipynb`, README* |
| *Pedro Garcia Pintor* | *`02_caracterizacao.ipynb`, `03_baseline.ipynb`* |

## Declaração de uso de IA generativa

O Claude Code (Anthropic) foi usado para: auditar o protótipo existente contra a rubrica
do enunciado, o split treino/val/teste por bloco espacial dentro do `notebooks/01_dados.ipynb`, ajudar prototipar os notebooks de `01_dados.ipynb`,`02_caracterizacao.ipynb`, `03_baseline.ipynb`. 

## Discussão


**Problema.** Estimar a qualidade de pastagem — usando como proxy o vigor MapBiomas
(baixo/médio/alto) — a partir de imagens de satélite, numa região do Brasil onde as bases
de imagem sugeridas pelo enunciado (EuroSAT, UC Merced) não se aplicam por não cobrirem o
Brasil. Como não existe base pública rotulada diretamente por "qualidade de grama", o
rótulo de vigor do MapBiomas é adotado explicitamente como proxy (opção (b) do enunciado,
adaptada: discretização de um índice de vegetação por um produto de terceiros em vez de
por quantis próprios). O objetivo mais amplo do projeto é identificar qualidade de grama
de forma geral, não apenas pastagem de pecuária; neste protótipo inicial, porém, o escopo
foi restringido a pastagem (classe MapBiomas 15) como ponto de partida de elaboração, já
que o MapBiomas fornece um rótulo proxy (vigor) e uma máscara de classe prontos para uso
imediato. Um próximo passo natural é substituir/complementar essa fonte por outras classes
de cobertura herbácea do MapBiomas (por exemplo, formação campestre, vegetação herbácea
urbana) e agregar tudo em um único dataset unificado de "grama", ampliando o escopo além
da pastagem de pecuária.

**Dados.** Duas fontes públicas via Google Earth Engine: **MapBiomas Coleção 9** (ativo
Pasture Vigor para o rótulo + LULC para a máscara de classe Pastagem, código 15; uso
público com atribuição, brasil.mapbiomas.org) e **Sentinel-2 L2A harmonizado**
(`COPERNICUS/S2_SR_HARMONIZED`, ESA/Copernicus, acesso aberto). Período: anos de vigor
2019–2022 (4 anos), com composição sazonal Sentinel-2 de mediana (jun–set, estação seca em
Goiás, filtro `CLOUDY_PIXEL_PERCENTAGE < 20`) de cada ano correspondente. Amostragem
estratificada de 150 pontos por classe de vigor por ano (4 anos × 3 classes × 150 = 1.800
pontos) sobre a mesma AOI de ~65 km × 65 km perto de Goiânia-GO usada no modelo bônus.
Para cada ponto extrai-se um recorte fixo de 64×64 px a 10 m/pixel (bandas B2/B3/B4/B8/
B11/B12). **Tratamento de ausentes:** pixels sem observação válida no composto recebem
`defaultValue=0`; patches com mais de 5% de pixels zerados seriam descartados — na
prática, os 1.800 pontos amostrados passaram integralmente nessa checagem (nenhum
descartado por nuvem/dado ausente). **Duplicatas:** checadas por (lon, lat, ano de vigor);
nenhuma encontrada. Dataset final versionado em `data/processed/s2_patches.npz`:
X ∈ R^(1800×64×64×6), y balanceado 600/600/600 entre baixo/médio/alto vigor.
**Correção de tratamento:** a extração original amostrava o mosaico Sentinel-2 sem fixar
sua projeção/escala nativa antes do `sampleRectangle`, que acabava amostrando numa grade
padrão bem mais grosseira que 10 m — cada patch virava, na prática, um único valor de
pixel replicado por toda a área 64×64 (quase sem variação espacial real). Corrigido
reprojetando o mosaico para `EPSG:4326` a 10 m/pixel antes da extração, verificado por um
smoke test em 2 pontos (desvio espacial > 0 e centenas de valores únicos por banda) antes
de repetir a extração completa dos 1.800 pontos; os números acima já refletem essa versão
corrigida.

**Caracterização.** As 3 classes de vigor são balanceadas por construção (600/600/600 no
total) e nenhum patch foi perdido na checagem de qualidade de `01_dados.ipynb` (100%
aprovados), então o tratamento de dados não introduz desbalanceamento. A distribuição por
*partição*, porém, não é uniforme — reflexo do split por bloco espacial, não da classe:
teste tem 111 "Alto vigor" / 86 "Médio vigor" / 65 "Baixo vigor"; treino 402/402/448;
validação 87/112/87 (mesma ordem). Estatísticas por banda (média ± desvio, mín.–máx.): B2
555,6±188,7 (53–5448), B3 873,5±260,1 (113–5736), B4 1252,0±460,4 (80–6048), B8
2543,3±472,1 (59–6533,5), B11 3081,8±707,8 (125–6759), B12 2066,1±581,9 (86–6826) — o
intervalo mín.–máx. bem mais largo que média±3σ em todas as bandas sugere pixels de
outlier residuais (possível nuvem/sombra/brilho) mesmo após o filtro de <5% de pixels
zerados por patch. Por classe, B8 (NIR) e as bandas SWIR (B11/B12) já mostram alguma
separação fisicamente esperada: "Alto vigor" tem a maior média de B8 (2682 vs. 2405 em
"Baixo vigor") e as menores médias de B11/B12 — padrão consistente com dossel vegetal mais
denso/menos solo exposto em vigor mais alto, e coerente com a regressão logística
conseguir separar as 3 classes razoavelmente bem (ver Baseline e resultados). NDVI médio
geral: 0,355 (desvio 0,076) — típico de pastagem, com sobreposição visível entre classes
no histograma (`02_caracterizacao.ipynb`), sem um limiar único que as separe claramente.
Sazonalidade controlada por desenho: todos os compostos usam a mesma janela jun–set
(estação seca) do ano do ponto, o que deve reduzir o efeito de variação sazonal entre os 4
anos amostrados (2019–2022) — não testado estatisticamente aqui.

**Protocolo experimental.** Split 70/15/15 por **bloco espacial de grade** (~5,5 km,
`GRID_DEG=0,05`, bem maior que o recorte de 640 m): cada ponto é atribuído a uma célula de
grade e **blocos inteiros** — não pontos individuais — são alocados a uma única partição
por um algoritmo guloso que mira as proporções 70/15/15, com verificação explícita
(`assert`) de que nenhum bloco aparece em mais de uma partição. Semente fixa (42) na
amostragem e no split. Resultado real: 1.252 treino / 286 validação / 262 teste (69,6% /
15,9% / 14,6% — o pequeno desvio do alvo exato vem de alocar blocos inteiros, não pontos,
a cada partição).

**Baseline e resultados.** No conjunto de teste, o baseline trivial (classe majoritária)
obteve acurácia 0,248 e F1 macro 0,133. A regressão logística sobre atributos agregados
por recorte (médias e desvios por banda + NDVI/GNDVI, 15 atributos; validação: acurácia
0,671, F1 macro 0,674) obteve acurácia 0,641 e F1 macro 0,630 no teste — mais que o dobro
do trivial em acurácia e quase 5× em F1 macro. O `classification_report` mostra
desempenho equilibrado entre as 3 classes, sem colapso em nenhuma: Baixo vigor (precisão
0,62 / recall 0,69 / F1 0,66, suporte 65), Médio vigor (0,52 / 0,47 / 0,49, suporte 86),
Alto vigor (0,73 / 0,75 / 0,74, suporte 111) — essa última é, na verdade, a classe mais
bem discriminada. No total, 94 dos 262 exemplos de teste (35,9%) foram classificados
incorretamente. Esse resultado é bem mais forte que numa versão anterior deste pipeline
(acurácia de teste ~0,275, com colapso total na classe "Alto vigor"), rodada antes da
correção da extração de patches Sentinel-2 descrita em Dados — evidência de que o ganho
veio de corrigir a captura de sinal espacial real nas bandas, não de mudança na amostragem
de pontos ou no rótulo.

**Limitações.** O rótulo de vigor é saída de um modelo, não verdade de campo (ver
[Limitações conhecidas / próximos passos](#limitações-conhecidas--próximos-passos) acima).
Os recortes são pontos isolados no tempo/espaço, não agregados por propriedade. O NDVI
médio por classe se sobrepõe bastante (a Caracterização mostra histogramas sem um limiar
único que separe as 3 classes), e o intervalo mín.–máx. das bandas sugere outliers
residuais (possível nuvem/sombra) que passam pelo filtro de <5% de pixels zerados — uma
checagem de qualidade mais rígida (ex. por desvio por banda, não só fração de zeros)
poderia ser um refinamento futuro.

**Trabalhos futuros.** Três direções (ver também a lista do modelo bônus acima): (1) usar
ou combinar pontos validados em campo pelo LAPIG para avaliação, já que o rótulo de vigor
do MapBiomas é ele próprio saída de um modelo; (2) comparar este baseline raso de imagem
bruta com o LSTM sobre embeddings AlphaEarth nos *mesmos* pontos, como uma ablação
controlada de embeddings vs. pixels brutos — agora que o baseline raso captura sinal
espacial real, essa comparação fica mais informativa; (3) agregar sobre polígonos de
propriedades do CAR em vez de pontos, assim que houver limites disponíveis.
