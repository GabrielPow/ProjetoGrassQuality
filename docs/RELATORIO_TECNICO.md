# Detecção não supervisionada de perturbações na Amazônia com embeddings do TerraMind e Transformer autoencoder temporal

**Relatório técnico — IBM8924 Projeto de Deep Learning, Grupo 1, 2026.2**
Gabriel Pow, Pedro Pintor · novembro de 2026

---

## Resumo

Desmatamento, garimpo e fogo alteram a Amazônia a cada trimestre, mas os sistemas oficiais (DETER e PRODES)
chegam com atraso e não enxergam eventos pequenos. Este trabalho investiga se um modelo que aprende apenas como um lugar *normal* evolui consegue
apontar perturbações sem nunca ver um exemplo rotulado, e se isso funciona melhor que métodos espectrais simples.
Montamos um cubo trimestral de Sentinel-2 e Sentinel-1 (2019–2025) sobre o Tapajós (PA), extraímos embeddings
do *foundation model* TerraMind Small congelado em células de 320 m e treinamos um Transformer autoencoder
temporal que reconstrói um trimestre escondido a partir de oito trimestres de contexto; a nota de anomalia é o
erro de reconstrução. O trabalho junta duas implementações do grupo: as 20 features espectrais clássicas da
implementação original (bandas, NDVI, GNDVI, BSI e energia de Fourier) foram levadas para a mesma grade e
combinadas com o autoencoder por fusão de scores e por entrada conjunta. As regras de avaliação (divisão
espacial e temporal, prevalência real, bootstrap por blocos) foram fixadas antes dos resultados. Num teste rápido com embeddings AlphaEarth,
a mudança do embedding superou a queda de NDVI (AUROC 0,89 × 0,82). Na sub-área processada (256 patches, 29
eventos no teste limitado), porém, a queda de NDVI obteve AUROC 0,77 contra 0,67 do autoencoder, com intervalos
sobrepostos; no limiar de 1% de falso alarme, o autoencoder não marcou nenhum evento. A fusão ajudou no
diagnóstico (0,68 → 0,72) e não no teste limitado. Concluímos que a pergunta central segue em aberto e que o
teste nas áreas completas, já preparado, é o próximo passo decisivo.

---

## 1. Introdução

### 1.1 Problema

A Amazônia brasileira perde floresta todos os anos por corte raso, degradação, fogo e garimpo. O Brasil tem dois
sistemas oficiais de monitoramento por satélite, mantidos pelo INPE: o **PRODES**, que mede o desmatamento anual
por corte raso, e o **DETER**, que emite alertas quase diários para fiscalização. Ambos dependem de interpretação
visual de especialistas e têm limitações conhecidas: o PRODES é anual; o DETER tem área mínima de mapeamento de
cerca de 3 ha e não registra boa parte dos cortes pequenos (no nosso levantamento, só 37% dos patch-anos com PRODES
no Tapajós têm alerta DETER); e ambos procuram classes predefinidas.

Métodos supervisionados de aprendizado profundo herdam essas limitações, porque aprendem a reproduzir os rótulos
existentes. Rótulos de perturbação são incompletos, chegam atrasados e não cobrem tipos novos (um garimpo que muda
de forma, um padrão de degradação não catalogado).

### 1.2 Motivação

A **detecção de anomalias** inverte a lógica: em vez de aprender o que é cada evento, o modelo aprende como um lugar
normal evolui ao longo do tempo (incluindo a sazonalidade da estação seca) e marca o que foge disso. Assim, o
modelo não depende de rótulos de evento para treinar, e os rótulos existentes servem só para avaliar.

Ao mesmo tempo, *foundation models* de observação da Terra, como o **TerraMind** (IBM + ESA) e o **AlphaEarth**
(Google), oferecem representações pré-treinadas em milhões de imagens que, em princípio, capturam mais que índices
espectrais feitos à mão. Faltava testar se essas representações, analisadas no tempo, de fato ajudam a detectar
perturbações sem rótulos.

### 1.3 Objetivos

**Pergunta central.** Embeddings de um *foundation model* (TerraMind), analisados ao longo do tempo, detectam
perturbações sem rótulos melhor do que features espectrais clássicas e do que o mesmo embedding sem histórico?

A pergunta foi decomposta numa sequência de perguntas menores, cada uma motivada pela anterior:

| | Pergunta | Hipótese |
|---|---|---|
| P0 | Dá para achar perturbação sem ensinar o que ela é? | Embeddings prontos separam perturbação de normal melhor que o NDVI (H1) |
| P1 | Que dado e que tamanho de unidade usar? | Células pequenas (320 m) detectam melhor que patches inteiros (H4) |
| P2 | Como avaliar sem trapacear? | — (protocolo) |
| P3 | Como a IA dá uma nota de anomalia? | — (arquitetura) |
| P4 | Um método simples já resolve? | O Transformer AE supera os baselines simples (H1) |
| P5 | Olhar o histórico ajuda? | Comparar com o passado detecta melhor que olhar só o presente (H2) |
| P6 | Juntar features clássicas e TerraMind ajuda? | A combinação supera cada parte (H3) |

### 1.4 Contexto: as duas implementações

O projeto nasceu como classificação de **qualidade de pastagem** (escopo do Grupo 1 da AC), com rótulo *proxy* de
vigor do MapBiomas perto de Goiânia. Essa trilha (implementação original, `notebooks/01`–`07`) produziu o pipeline
de ingestão Sentinel-2, as features espectrais e de textura e um baseline de regressão logística (seção 5.6). Em
paralelo, uma segunda trilha (`notebooks/gabriel/`) passou à detecção de anomalias na Amazônia com TerraMind.
A versão final junta as duas: as features da implementação original viram um segundo sinal para o modelo da
segunda trilha (seção 4.6).

---

## 2. Revisão bibliográfica

### 2.1 Monitoramento de desmatamento na Amazônia

O **PRODES** (INPE) mapeia anualmente o corte raso da floresta primária desde 1988, com imagens Landsat e
interpretação visual; é a estatística oficial do desmatamento. O **DETER** (Diniz et al., 2015) emite alertas
quase diários com sensores de resolução mais baixa (WFI/CBERS-4, AWiFS), cobrindo também degradação, mineração e
cicatrizes de fogo. **Hansen et al. (2013)** produziram mapas globais anuais de perda de cobertura arbórea a
30 m. O **MapBiomas** (Souza et al., 2020) reconstrói anualmente o uso e cobertura do solo de todo o Brasil desde
1985. Para fogo, o produto **MODIS MCD64A1** (Giglio et al., 2018) mapeia área queimada mensal a 500 m. Usamos
todas essas fontes apenas como rótulos de avaliação.

### 2.2 Detecção de mudança em séries temporais de satélite

Os métodos clássicos de detecção de mudança modelam a série temporal de cada pixel e marcam rupturas: **BFAST**
(Verbesselt et al., 2010) decompõe a série em tendência, sazonalidade e resíduo; **LandTrendr** (Kennedy et al.,
2010) segmenta trajetórias anuais do Landsat; **CCDC** (Zhu e Woodcock, 2014) ajusta modelos harmônicos e
detecta desvios contínuos. Alertas com radar, como o **RADD** (Reiche et al., 2021), usam Sentinel-1 para
enxergar através das nuvens. Esses métodos dependem de índices espectrais escolhidos à mão (NDVI, NBR) e de
limiares calibrados por região.

### 2.3 Índices espectrais

O **NDVI** (Rouse et al., 1974) mede o vigor da vegetação pela diferença normalizada entre infravermelho próximo e
vermelho. O **GNDVI** (Gitelson et al., 1996) troca o vermelho pelo verde e é mais sensível à clorofila. O **BSI**
(Rikimaru et al., 2002) realça solo exposto, útil para desmatamento recente e garimpo. A análise de textura por
transformada de Fourier, explorada na implementação original, mede a energia em diferentes frequências espaciais
e capta mudança de padrão (por exemplo, dossel contínuo → clareira).

### 2.4 Foundation models de observação da Terra

O **TerraMind** (Jakubik et al., 2025) é um modelo generativo multimodal pré-treinado pela IBM e pela ESA em
dados de Sentinel-1, Sentinel-2, modelos de elevação, cobertura do solo e texto, com pesos abertos. O
**AlphaEarth Foundations** (Brown et al., 2025), do Google, gera um embedding anual de 64 dimensões por pixel de
10 m, disponível como coleção no Earth Engine, mas sem acesso ao modelo ou à entrada. Ambos mostram bons
resultados em tarefas supervisionadas com poucos rótulos; seu uso para detecção não supervisionada de mudança é
menos estudado.

### 2.5 Detecção de anomalias com aprendizado profundo

A detecção de anomalias (Chandola et al., 2009) aprende a distribuição dos dados normais e marca o que se desvia.
Métodos rasos incluem o **kNN** (distância aos vizinhos normais) e o **Isolation Forest** (Liu et al., 2008).
O **Deep SVDD** (Ruff et al., 2018) aprende uma hiperesfera mínima que envolve os normais. **Autoencoders**
treinados só com normais usam o erro de reconstrução como nota de anomalia; em séries temporais, a versão com
LSTM (Malhotra et al., 2016) e, mais recentemente, com **Transformer** (Vaswani et al., 2017) modelam a
dependência temporal. A formulação *masked*, inspirada em He et al. (2022), esconde partes da entrada durante o
treino e força o modelo a prever o que falta em vez de copiar.

### 2.6 Avaliação com dados espaciais e eventos raros

Validação cruzada aleatória em dados espaciais superestima o desempenho porque vizinhos quase idênticos caem em
treino e teste; Roberts et al. (2017) recomendam blocos espaciais. Com eventos raros, a acurácia é enganosa e a
curva precisão-recall é mais informativa que a ROC (Saito e Rehmsmeier, 2015).

---

## 3. Dados

### 3.1 Áreas de estudo

| | Área principal | Área de generalização |
|---|---|---|
| Região | Tapajós / BR-163 (PA) | Norte de Rondônia |
| Caixa (lon, lat) | −57 a −55, −7,5 a −5,5 | −64 a −62, −10 a −8 |
| Projeção | UTM 21S | UTM 20S |
| Patches de 2,24 km | 9.801 | 9.644 (só teste) |
| Papel | treino, validação e teste | só teste: outro estado, outro histórico de ocupação |

Rondônia foi escolhida por ter o maior número de alertas DETER de desmatamento entre as candidatas (11,4 mil desde
2019). Ela quase não tem garimpo (29 alertas), então mineração só é avaliada no Tapajós.

**Sub-área de desenvolvimento.** Um recorte de 16 × 16 patches (35,8 km de lado, 256 patches) do Tapajós foi
escolhido por ter muitos eventos e serve para desenvolvimento. Todos os números deste relatório vêm dela (ver
limitações, seção 6.3).

### 3.2 Imagens

- **Sentinel-2 L2A** (Drusch et al., 2012): 12 bandas ópticas, 10–60 m, reamostradas para 10 m.
- **Sentinel-1 GRD** (Torres et al., 2012): radar de abertura sintética (VV, VH), que atravessa nuvens.
- Período: 2019–2025, **28 trimestres**. Cada trimestre é a mediana das passagens sem nuvem (máscara de nuvem do
  produto L2A).
- Grade: patches de **224 × 224 pixels a 10 m (2,24 km)**, alinhados a múltiplos de 2.240 m em UTM e agrupados em
  blocos de 4 × 4 patches (8,96 km).
- Trimestres sem nenhuma observação limpa (comuns de dezembro a abril) recebem uma marca explícita de "sem dado"
  no modelo, em vez de zeros.
- Acesso via Google Earth Engine (Gorelick et al., 2017). O cubo da sub-área ocupa ~5–8 GB (Zarr).

### 3.3 Rótulos (só para avaliação)

O modelo **nunca vê rótulos de evento**. Eles servem apenas para medir o desempenho.

| Fonte | Tipos | Data usada |
|---|---|---|
| DETER | desmatamento, mineração, fogo, degradação, corte seletivo | data do alerta |
| PRODES | desmatamento | janela agosto/(Y−1) até a data da imagem |
| Hansen GFC `lossyear` | desmatamento (checagem cruzada) | ano civil |
| MapBiomas LULC | desmatamento (floresta → uso antrópico), mineração (entrada na classe 30) | ano civil |
| MODIS MCD64A1 | fogo, **só sobre floresta natural** (fogo em pastagem é manejo) | trimestre |

Para cada patch e período, a fração do patch tocada por cada fonte é somada (limite 1), e cada tipo de evento
fica com o máximo entre as fontes. Definições:

- **Evento:** fração ≥ 1% do patch (≈ 5 ha).
- **Ambíguo:** 0 < fração < 1%; excluído da avaliação.
- **Limpo:** nenhuma fonte toca o patch no período.
- **Rótulo trimestral:** só fontes com data sub-anual (DETER, MCD64A1). É o mais exigente e o mais próximo de um
  sistema de alerta.

**Normais para treino.** Uma amostra (patch, trimestre do ano T) é normal se os anos T−1, T e T+1 não têm evento em
nenhuma fonte. A regra original (soma 2019–2025 abaixo de 0,1% do patch) deixava 2.008 patches de floresta e
**apenas 1 de pastagem** no treino do Tapajós, porque um patch de pastagem de 2,24 km quase sempre tem remanescentes
de floresta sendo abertos. A regra por janela, registrada antes de qualquer resultado, dá 11.360 janelas de
floresta e 544 de pastagem.

**Prevalência.** Com todas as fontes, ~23–29% dos patch-anos e ~4–5% dos patch-trimestres têm evento no período de
teste do Tapajós. Na sub-área, ~5–7% dos patch-trimestres têm evento.

### 3.4 Dados da implementação original (pastagem)

- 1.800 pontos do MapBiomas *Pasture Vigor* (450 por ano, 2019–2022, balanceados entre vigor baixo, médio e alto)
  numa área de ~4.977 km² perto de Goiânia (GO), restritos a pixels de pastagem (classe 15 do MapBiomas LULC).
- Recorte Sentinel-2 de **64 × 64 pixels, 6 bandas** (B2, B3, B4, B8, B11, B12) por ponto:
  tensor **X ∈ ℝ^(1800 × 64 × 64 × 6)**. Todos os 1.800 recortes passaram na checagem de qualidade (≤ 5% de pixels
  sem dado).
- Divisão 70/15/15 por bloco de grade espacial (154 blocos), semente 42: 1.252 / 286 / 262.

---

## 4. Metodologia

### 4.1 Visão geral do pipeline

```
Sentinel-2 + Sentinel-1 (Earth Engine)
  → mediana trimestral, patch 224 × 224 px, 28 trimestres
  → TerraMind Small congelado ──────────→ 384 números por célula de 320 m (49 células/patch)
  → features clássicas por célula ──────→ 20 números por célula
  → Transformer autoencoder temporal (só normais): 8 trimestres de contexto → reconstrói o 9º
  → nota da célula = erro de reconstrução; nota do patch = máximo das 49 células
  → avaliação contra DETER/PRODES/Hansen/MapBiomas/MODIS (só aqui os rótulos entram)
```

### 4.2 P0: teste rápido com AlphaEarth

Antes de construir qualquer modelo, usamos embeddings prontos do AlphaEarth (1 por ano, 64 dimensões, pixel de
10 m) em 2.400 pontos do Tapajós (671 de teste), com rótulos do MapBiomas e nenhum modelo treinado. Três notas de
anomalia foram comparadas: queda de NDVI; distância do embedding atual à média dos 2 anos anteriores (com
histórico); e distância do embedding atual aos normais (sem histórico). Código em
`experiments/amazon_anomaly_firstpass/`.

### 4.3 P1: unidade de análise

O TerraMind recebe 224 × 224 pixels e devolve 196 tokens (grade 14 × 14) de 384 dimensões. A média de cada bloco de
2 × 2 tokens forma a ficha de uma **célula de 320 m** (49 por patch). Comparamos a nota no patch inteiro e em
recortes maiores com a nota por célula. No único tamanho comparável (patch de 2,24 km), as alternativas
empataram (0,57 × 0,54), e o desempenho piorou conforme o pedaço crescia; num teste inicial, a mesma nota deu
AUROC 0,57 na média do patch contra 0,84 no token. Por isso a nota é calculada por célula e o patch
recebe a **maior** nota das 49 células. Um garimpo típico ocupa ≈ 5% do patch e quase some na média.

### 4.4 Peça 1: o TerraMind como extrator de features

- **Modelo:** TerraMind v1 Small (22 M de parâmetros), **congelado**. Sem GPU e sem rótulos suficientes, não faz
  sentido ajustar um modelo desse porte; com pesos abertos, controlamos a entrada (o que o AlphaEarth não permite).
- **Entrada:** 12 bandas do Sentinel-2 + VV/VH do Sentinel-1. A combinação S2 + S1 foi melhor que só S2 em todos os
  métodos baseados em TerraMind no diagnóstico.
- **Custo:** 40 ms/patch (S2) e 81 ms/patch (S2 + S1) em CPU.

### 4.5 A alternativa clássica: 20 features espectrais por célula

Herdadas da implementação original (`dataset_utils.py`, `07_fourier_mudanca.ipynb`) e recalculadas por célula de
320 m em `features_originais.py`:

| Grupo | Features |
|---|---|
| Nível | médias de B2, B3, B4, B8, B11, B12 |
| Vegetação | média e desvio do NDVI; média do GNDVI |
| Solo exposto | média do BSI |
| Textura | desvio de cada uma das 6 bandas; log10 da energia de Fourier do NDVI (janela de Hann) em 4 anéis de frequência: (0; 0,04], (0,04; 0,08], (0,08; 0,125] e > 0,125 ciclo/pixel |

Reflectância = DN / 10.000. Células com menos de 50% de pixels limpos ficam sem dado. Na implementação original, a
transformada de Fourier da imagem inteira não localizava mudança (AUROC ≈ 0,5); só funcionou localmente, por célula.

### 4.6 Peça 2: Transformer autoencoder temporal

- **Tarefa:** recebe 9 trimestres consecutivos de fichas de uma célula, com o 9º **escondido**, e reconstrói o 9º.
  Se o modelo visse o 9º, poderia simplesmente copiar a anomalia.
- **Formulações:** *forecast* (só esconde o último) e *masked* (no treino, também esconde ~25% do contexto). A
  *masked* foi escolhida pela perda nos normais de validação (0,410 × 0,422), sem olhar rótulos.
- **Arquitetura** (`modelo.py`): projeção linear da entrada para d = 128; codificação do ano e do trimestre por
  uma pequena MLP somada a cada passo (a estação seca é normal); um token aprendido substitui os passos escondidos
  e os trimestres sem dado; 4 camadas de Transformer encoder (4 cabeças, *feed-forward* 512, *dropout* 0,1);
  gargalo de 64 dimensões; cabeça linear de volta à dimensão da entrada. 0,9 M de parâmetros.
- **Treino:** só sequências normais de blocos de treino que terminam até 2022 (26.656 sequências); AdamW
  (lr = 10⁻³, *weight decay* 10⁻⁴), agendamento cosseno, lotes de 512, até 40 épocas com parada antecipada
  (paciência 6) pela perda nos normais de validação de 2021–2022; treina em CPU.
- **Nota de anomalia:** erro quadrático médio da reconstrução do trimestre escondido.

### 4.7 Baselines

| Grupo | Método | Ideia |
|---|---|---|
| Sem IA | Queda de NDVI | NDVI histórico (média dos trimestres anteriores válidos) − NDVI atual |
| TerraMind + uma conta | kNN no delta | distância do delta (ficha atual − histórico) aos k = 10 vizinhos normais |
| | kNN estático | distância da ficha atual aos normais, sem histórico |
| | Cosseno vs. histórico | 1 − similaridade entre ficha atual e histórico |
| Rede pequena / raso | Isolation Forest (300 árvores), Deep SVDD | sobre `[F, F − histórico]` das features TerraMind ou clássicas |
| Textura | Mudança de Fourier | soma de (log E(t) − log E_hist)² nos anéis de alta frequência |

Todos ajustados só com células normais de treino.

### 4.8 A junção das duas implementações (P6)

Dois caminhos, ambos registrados no protocolo antes dos resultados:

- **A. Fusão de scores.** Para cada método, `z = (score − mediana) / IQR`, com mediana e IQR dos scores dos normais
  de validação. Nota combinada = média dos z. A combinação principal, fixada antes, é **AE (TerraMind) + Isolation
  Forest nas 20 features clássicas**. Outras combinações e regras (média e máximo de percentis) são ablação.
- **B. AE com entrada combinada.** Mesma arquitetura, entrada de 404 dimensões (384 TerraMind + 20 clássicas,
  padronizadas pelos normais de treino). Perda e nota = 0,5 · MSE(TerraMind) + 0,5 · MSE(clássicas), para que as
  384 dimensões não abafem as 20. Treinamos três AEs: só TerraMind, só clássicas e combinado.

Descartamos dois caminhos: um classificador supervisionado sobre as notas (violaria o protocolo, que só permite
rótulos na avaliação) e aplicar o AE à tarefa de pastagem de Goiânia (sem série temporal, recortes de 64 px e 6
bandas, e um AE que só viu floresta amazônica).

### 4.9 Protocolo de avaliação (P2)

Fixado em `notebooks/gabriel/PROTOCOLO.md` em 15/10/2026, antes de qualquer treino; mudanças posteriores foram
registradas com data e motivo, sempre antes de ver resultados.

- **Divisão espacial:** blocos de 4 × 4 patches (8,96 km) inteiros vão para treino, validação ou teste (70/15/15,
  semente 42). Nenhum bloco aparece em duas partições (verificado por `assert`).
- **Divisão temporal:** treino com sequências até 2022; avaliação em 2023–2025.
- **Ajustes** (hiperparâmetros, parada, limiar, normalização da fusão) só com amostras normais de validação.
- **Prevalência real:** positivos = patch-trimestres com evento; negativos = todos os limpos. Sem rebalancear.
- **Métricas:** AUROC e AUPRC com IC 95% por bootstrap de blocos espaciais (300 reamostragens); TPR com 1% e 5% de
  FPR; precisão@k. Comparações entre métodos com bootstrap pareado (mesmos blocos para os dois métodos).
- **Limiar operacional:** quantil dos scores limpos de validação (2021–2022) para FPR = 1%.

**Por que não acurácia?** Só ~5% dos patch-trimestres têm evento: dizer "tudo normal" já acerta ~95%.

**Teste limitado.** As áreas completas não foram processadas, então o teste oficial (blocos de teste do Tapajós e
Rondônia inteira) ainda não existe. Os blocos de teste da sub-área não têm nenhum evento em 2023–2025. Registramos
em 05/11 (seção 7 do protocolo), antes de calcular, que o teste limitado seria feito nos **blocos de validação da
sub-área em 2023–2025**: 29 patch-trimestres com evento e 537 limpos, separados no espaço (nenhum modelo treinou
neles) e no tempo (2023–2025 nunca foi visto). Os números marcados como **diagnóstico** vêm dos blocos de treino
em 2023–2025 (166 eventos), separados só no tempo.

| Partição | Período | Eventos | Limpos | Ambíguos |
|---|---|---|---|---|
| treino | 2019–2022 | 213 | 2.800 | 59 |
| treino (diagnóstico) | 2023–2025 | 166 | 2.100 | 38 |
| validação | 2019–2022 | 82 | 676 | 10 |
| **validação (teste limitado)** | **2023–2025** | **29** | **537** | 10 |
| teste | 2019–2022 | 0 | 256 | 0 |
| teste | 2023–2025 | 0 | 192 | 0 |

---

## 5. Resultados

### 5.1 P0: embeddings prontos funcionam sem rótulos

| Nota de anomalia (AlphaEarth, Tapajós, 671 pontos de teste) | AUROC (IC 95%) |
|---|---|
| Clássico: queda de NDVI | 0,82 (0,78–0,85) |
| AlphaEarth com histórico: mudança em relação aos 2 anos anteriores | 0,89 (0,86–0,92) |
| AlphaEarth sem histórico: distância aos normais | **0,93** (0,90–0,95) |

O embedding pronto superou o NDVI, mas o método sem histórico venceu: ele reconhece *onde* está a fronteira de
desmatamento, não *quando* algo mudou. Além disso, a amostra era meio a meio (51% de eventos, contra < 1% no mundo
real), e MapBiomas e AlphaEarth vêm de satélites parecidos, então rótulo e modelo não são independentes. Esses três
alertas motivaram P2 e P5.

### 5.2 P3: treino do Transformer AE

A formulação *masked* teve perda menor nos normais de validação que a *forecast* (0,410 × 0,422). As curvas de
perda das três entradas (TerraMind, clássicas, combinada) estão em `results/apresentacao/11_curvas.png`; a entrada
combinada começa a sobreajustar mais cedo, e a parada antecipada guarda a melhor época.

### 5.3 P4: comparação com os métodos simples (teste limitado)

| Método | AUROC | IC 95% | AUPRC | TPR @ 1% FPR | Desmat. (8) | Fogo (17) | Garimpo (9) |
|---|---|---|---|---|---|---|---|
| Queda de NDVI | **0,77** | 0,63–0,82 | 0,16 | 0,00 | 0,73 | **0,90** | **0,65** |
| kNN no delta (TerraMind) | 0,74 | 0,63–0,82 | **0,25** | **0,07** | 0,78 | 0,89 | 0,53 |
| Deep SVDD (features clássicas) | 0,69 | 0,48–0,72 | 0,11 | 0,00 | **0,80** | 0,78 | 0,54 |
| Fusão de scores (AE + IForest clássico) | 0,68 | 0,61–0,76 | 0,08 | 0,00 | 0,75 | 0,77 | 0,53 |
| AE (TerraMind) | 0,67 | 0,58–0,74 | 0,09 | 0,00 | 0,75 | 0,77 | 0,49 |
| AE (features clássicas) | 0,66 | 0,53–0,80 | 0,08 | 0,03 | 0,68 | 0,73 | 0,60 |
| AE (TerraMind + clássicas) | 0,63 | 0,52–0,77 | 0,08 | 0,03 | 0,66 | 0,71 | 0,57 |

AUPRC ao acaso ≈ 0,05. Um patch-trimestre pode ter mais de um tipo de evento.

**No diagnóstico** (blocos de treino, 166 eventos) a ordem muda: Deep SVDD clássico 0,74; kNN no delta 0,73; AE
clássico 0,72; AE combinado 0,72; queda de NDVI 0,71; fusão 0,71; AE TerraMind 0,68. A instabilidade da ordem entre
as duas avaliações é coerente com os intervalos largos.

**No limiar de alarme** (AE combinado, limiar calibrado nos limpos de validação de 2021–2022):

| Limiar | Eventos marcados | Alarmes falsos | Eventos perdidos | Acurácia | Acurácia "tudo normal" | Precisão | Recall |
|---|---|---|---|---|---|---|---|
| 1% FPR | 0 de 29 | 2 | 29 | 0,945 | 0,949 | 0,000 | 0,000 |
| 5% FPR | 1 de 29 | 11 | 28 | 0,931 | 0,949 | 0,083 | 0,034 |

Com 1% de alarme falso, o modelo não marca nenhum evento e tem acurácia abaixo de dizer "tudo normal". Nas curvas
ROC, no trecho de poucos alarmes falsos, quem sobe primeiro são o kNN no delta e a queda de NDVI; os AEs ficam
perto do acaso.

### 5.4 P5: o histórico ajuda?

AUROC no diagnóstico, fichas TerraMind S2 + S1:

| Nota | Por ano | Por trimestre |
|---|---|---|
| Sem histórico: kNN estático | **0,97** | 0,71 |
| Com histórico simples: kNN no delta | 0,90 | **0,73** |
| Transformer AE (masked) | 0,79 | 0,68 |

Por ano, o método sem histórico vence pelo mesmo motivo do P0: reconhece a zona de fronteira. Por trimestre, que é o
cenário de um sistema de alerta, os métodos quase empatam e comparar com o passado de forma simples fica
ligeiramente à frente. O Transformer compara cada lugar com o próprio passado e, por isso, não usa o atalho da
fronteira; mas ainda não supera os métodos simples.

### 5.5 P6: juntar features clássicas e TerraMind ajuda?

| | Diagnóstico (166 eventos) | Teste limitado (29 eventos) |
|---|---|---|
| AE só TerraMind (384) | 0,68 | 0,67 |
| AE só features clássicas (20) | **0,72** | 0,66 |
| B. AE com as duas (404) | **0,72** | 0,63 |
| A. Fusão de scores (AE + IForest clássico) | 0,71 | **0,68** |

No diagnóstico, juntar ajudou (0,68 → 0,72), com ganho significativo em garimpo (+0,11, IC 95% 0,04 a 0,17, bootstrap
pareado). No teste limitado, o AE combinado ficou pior (0,63). Um achado lateral: no mesmo AE, as **20 features
clássicas empatam com as 384 dimensões do TerraMind** (0,72 × 0,68 no diagnóstico; 0,66 × 0,67 no teste limitado).

### 5.6 Trilha de pastagem (implementação original)

| Modelo | Acurácia | F1 macro |
|---|---|---|
| Baseline trivial (classe majoritária) | 0,248 | 0,133 |
| Regressão logística (15 atributos: média e desvio de 6 bandas, NDVI, GNDVI) | **0,641** | **0,630** |

No teste (262 recortes), o vigor alto teve o melhor F1 (0,74) e o médio o pior (0,49), coerente com a classe
intermediária se confundir com as vizinhas. Esses atributos foram a base das 20 features clássicas da seção 4.5.

### 5.7 Análise qualitativa

- **Mapa de notas** (2024T3, `16_mapa_anomalia.png`): a nota acende em eventos grandes de desmatamento e fogo.
- **Alarmes falsos:** parte deles é garimpo **visível** na imagem em patches que os rótulos chamam de limpos. Os
  rótulos têm lacunas, e parte do "erro" é acerto sem rótulo.
- **Eventos perdidos:** eventos de ~1% do patch (≈ 5 ha), no limite da definição de evento e menores que uma
  célula de 320 m.
- **Rondônia** (área nunca vista, ilustração escolhida a dedo, não métrica): a nota sobe nos três desmatamentos
  examinados, mas fica abaixo do limiar (nota ÷ limiar = 0,20, 0,43 e 0,25).

---

## 6. Discussão

### 6.1 Interpretação

**H1 (foundation model > NDVI).** Confirmada no P0 com AlphaEarth, mas não com TerraMind no teste limitado. A
diferença pode vir da tarefa (anual × trimestral), do desenho da amostra (balanceada × prevalência real) e da
dependência entre rótulo e modelo no P0. Com prevalência real e avaliação trimestral, o NDVI, que é barato e
interpretável, é um adversário forte para desmatamento e fogo. Ele falha onde o verde não cai: em garimpo, sua
AUROC cai para 0,65.

**H2 (histórico ajuda).** Por trimestre, um pouco. A comparação anual favorece métodos que reconhecem *onde* é a
fronteira, e isso infla métricas sem dizer nada sobre *quando* algo mudou. Essa é uma armadilha de avaliação que o
P0 revelou e que vale para trabalhos semelhantes.

**H3 (fusão ajuda).** Inconclusiva. O ganho no diagnóstico (principalmente em garimpo) não se repetiu no teste
limitado, e os intervalos se sobrepõem. O fato de 20 features clássicas empatarem com 384 dimensões do TerraMind
sugere que, nesta escala e com este modelo congelado, o embedding não carrega muito mais informação útil para
detecção de mudança do que índices espectrais bem escolhidos.

**H4 (célula pequena).** Confirmada. A nota por célula com máximo no patch é essencial para eventos pequenos.

**Por que o AE não supera?** Hipóteses: (i) o TerraMind foi pré-treinado para representar *o que* a cena é, não
para ser sensível a pequenas mudanças entre trimestres; (ii) o erro de reconstrução se espalha por 384 dimensões,
diluindo sinais concentrados em poucas; (iii) a sub-área tem só 256 patches e quase só floresta nos normais de
validação, pouco para o AE aprender a variabilidade normal; (iv) 320 m ainda é grande para eventos de poucos
hectares.

### 6.2 "Não ajudou" também é resultado

O protocolo fixado antes dos resultados impediu escolher a combinação vencedora olhando o teste. Isso tornou os
números menos favoráveis, mas confiáveis: o resultado negativo (o método simples empatou ou venceu) é uma
informação útil para quem pretende usar *foundation models* em alertas de desmatamento.

### 6.3 Limitações

1. **Áreas completas não processadas.** Feito: 256 patches (1,3% de 19.445) e 7.168 patch-trimestres (de ~545 mil),
   em ~1 h. O planejado exige 60–80 h de download do Earth Engine e ~19 h de TerraMind em CPU. Sem teste oficial
   nem de generalização.
2. **Sub-área escolhida pelos rótulos**, o que infla a prevalência; poucos eventos e intervalos largos.
3. **Blocos de validação também usados para ajuste** (com normais de 2021–2022), embora 2023–2025 nunca tenha sido
   visto.
4. **Normalização por cobertura** (pastagem × floresta) não testável: todos os normais de validação da sub-área são
   floresta.
5. **Infraestrutura:** sem GPU local, ~2 GB livres no Drive; o download do Earth Engine é o gargalo.
6. **Rótulos:** o DETER não vê cortes pequenos; o PRODES é anual; há eventos visíveis sem rótulo.
7. **Escala:** células de 320 m ainda são grandes para eventos de poucos hectares.

### 6.4 Trabalhos futuros

1. **Processar as áreas completas** com `stream_embeddings.py`, que já grava as features clássicas na mesma passada
   (~3 dias, retomável).
2. **Teste oficial** com amostra sorteada dos blocos de teste do Tapajós, sem olhar os rótulos antes.
3. **Rondônia completa** (generalização), com os três AEs, a fusão e os métodos simples.
4. **AE só com as 20 features clássicas** na área inteira: barato e não precisa do TerraMind.
5. **Tamanho da janela:** quanto histórico é preciso (2, 4, 8 ou 12 trimestres)?
6. **Normalização por cobertura** (pastagem × floresta).
7. **Mapas por trimestre e antecedência em relação ao DETER**; se os resultados se mantiverem, submissão ao
   IGARSS 2027.
8. Na trilha de pastagem: rótulo real de qualidade via campo (pontos validados do LAPIG), séries temporais de NDVI e
   agregação por imóvel (CAR).

---

## 7. Conclusão

Construímos um pipeline completo e reprodutível de detecção não supervisionada de perturbações na Amazônia, do
download de Sentinel-1/2 no Earth Engine à avaliação com cinco fontes de rótulo, e juntamos nele as duas
implementações do grupo: as features espectrais e de textura da trilha de pastagem e o Transformer autoencoder sobre
embeddings TerraMind.

O teste rápido com AlphaEarth mostrou que embeddings prontos detectam perturbação sem rótulos, mas também revelou
que avaliações anuais e balanceadas premiam quem reconhece a fronteira, não a mudança. Com um protocolo mais
exigente (trimestral, prevalência real, divisão espaço-tempo, regras fixadas antes), o Transformer AE ordena
eventos melhor que o acaso (AUROC 0,63–0,67), mas, na sub-área, não superou a queda de NDVI (0,77) nem o kNN no
delta (0,74), e no limiar operacional quase não marca eventos. Juntar features clássicas e TerraMind ajudou no
diagnóstico e não no teste limitado.

A pergunta central segue em aberto. Os 29 eventos do teste limitado não bastam para separar os métodos; o teste
nas áreas completas, cujo código já está pronto, decide.

---

## 8. Referências

- Brown, C. F. et al. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global
  mapping from sparse label data. *arXiv:2507.22291*.
- Chandola, V., Banerjee, A., Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1–58.
- Diniz, C. G. et al. (2015). DETER-B: The new Amazon near real-time deforestation detection system. *IEEE Journal
  of Selected Topics in Applied Earth Observations and Remote Sensing*, 8(7), 3619–3628.
- Drusch, M. et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services.
  *Remote Sensing of Environment*, 120, 25–36.
- Giglio, L. et al. (2018). The Collection 6 MODIS burned area mapping algorithm and product. *Remote Sensing of
  Environment*, 217, 72–85.
- Gitelson, A. A., Kaufman, Y. J., Merzlyak, M. N. (1996). Use of a green channel in remote sensing of global
  vegetation from EOS-MODIS. *Remote Sensing of Environment*, 58(3), 289–298.
- Gorelick, N. et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote
  Sensing of Environment*, 202, 18–27.
- Hansen, M. C. et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*,
  342(6160), 850–853.
- He, K. et al. (2022). Masked autoencoders are scalable vision learners. *CVPR*, 16000–16009.
- INPE. Programa de Monitoramento da Amazônia e Demais Biomas (PRODES e DETER). TerraBrasilis,
  http://terrabrasilis.dpi.inpe.br.
- Jakubik, J. et al. (2025). TerraMind: Large-scale generative multimodality for Earth observation.
  *arXiv:2504.11171*.
- Kennedy, R. E., Yang, Z., Cohen, W. B. (2010). Detecting trends in forest disturbance and recovery using yearly
  Landsat time series: 1. LandTrendr. *Remote Sensing of Environment*, 114(12), 2897–2910.
- Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation forest. *IEEE ICDM*, 413–422.
- Malhotra, P. et al. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. *arXiv:1607.00148*.
- Reiche, J. et al. (2021). Forest disturbance alerts for the Congo Basin using Sentinel-1. *Environmental
  Research Letters*, 16(2), 024005.
- Rikimaru, A., Roy, P. S., Miyatake, S. (2002). Tropical forest cover density mapping. *Tropical Ecology*,
  43(1), 39–47.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or
  phylogenetic structure. *Ecography*, 40(8), 913–929.
- Rouse, J. W. et al. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *Third ERTS Symposium*,
  NASA SP-351, 309–317.
- Ruff, L. et al. (2018). Deep one-class classification. *ICML*, 4393–4402.
- Saito, T., Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when
  evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
- Souza, C. M. et al. (2020). Reconstructing three decades of land use and land cover changes in Brazilian biomes
  with Landsat archive and Earth Engine. *Remote Sensing*, 12(17), 2735.
- Torres, R. et al. (2012). GMES Sentinel-1 mission. *Remote Sensing of Environment*, 120, 9–24.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*, 5998–6008.
- Verbesselt, J. et al. (2010). Detecting trend and seasonal changes in satellite image time series. *Remote
  Sensing of Environment*, 114(1), 106–115.
- Zhu, Z., Woodcock, C. E. (2014). Continuous change detection and classification of land cover using all
  available Landsat data. *Remote Sensing of Environment*, 144, 152–171.
