# Protocolo de avaliação (fixado em 15/10/2026, antes de qualquer treino)

Implementado em [rotulos.py](rotulos.py), [avaliacao.py](avaliacao.py) e [02_rotulos.ipynb](02_rotulos.ipynb);
parâmetros em [config.py](config.py). Mudar qualquer item abaixo depois de ver resultados de teste
exige registrar a mudança e o motivo aqui.

## 1. Áreas, período e unidade de análise

| | |
|---|---|
| AOI principal | Tapajós / BR-163 (PA), `[-57, -7,5, -55, -5,5]`, UTM 21S, 9.801 patches |
| AOI de generalização | Norte de Rondônia, `[-64, -10, -62, -8]`, UTM 20S, 9.644 patches, **só teste** |
| Período | 2019–2025, 28 trimestres (`2019Q1`…`2025Q4`) |
| Unidade | patch de 224 × 224 px a 10 m = **2240 m**, alinhado a múltiplos de 2240 m em UTM |

Rondônia foi escolhida por ser outro estado e outro histórico de ocupação, com o maior número de
alertas DETER entre as candidatas (11,4 mil de desmatamento desde 2019). Ela quase não tem garimpo
(29 alertas), então mineração só é avaliada no Tapajós.

## 2. Rótulos (só avaliação; o modelo não os vê)

| Fonte | Tipos | Data usada |
|---|---|---|
| DETER | desmatamento, mineração, fogo, degradação, corte seletivo | data do alerta |
| PRODES | desmatamento | janela ago/(Y-1) a data da imagem |
| Hansen GFC `lossyear` | desmatamento (checagem cruzada) | ano civil |
| MapBiomas LULC | desmatamento (floresta natural → uso antrópico), mineração (entrada na classe 30), água (ganho/perda da classe 33) | ano civil |
| MODIS MCD64A1 | fogo **só sobre floresta natural** (fogo em pastagem é manejo) | trimestre |

- **Fração do patch:** cada fonte é somada por patch-período (limite 1), e cada tipo fica com o máximo entre as fontes.
- **Evento:** fração ≥ 1% do patch (~5 ha). **Ambíguo:** 0 < fração < 1%; fica fora da avaliação. **Limpo:** nenhuma fonte toca o patch no período.
- **Rótulo anual** (patch × ano): todas as fontes. **Rótulo trimestral** (patch × trimestre): só fontes com data sub-anual (DETER, MCD64A1).
- **Normal de treino (regra em vigor desde 29/10): por janela.** Uma amostra (patch, trimestre t do
  ano T) é normal se os anos T-1, T e T+1 não têm evento (≥ 1%; ambíguo permitido) em nenhuma fonte
  (`rotulos.normal_windows`). O histórico de até 8 trimestres cabe em T-1..T, e o ano T+1 limpo evita
  treinar com a véspera de um evento.
  - *Regra anterior (15/10), por patch:* em cada fonte, a soma 2019–2025 ficava abaixo de 0,1% do patch.
    **Motivo da mudança** (registrado antes de qualquer resultado de modelo): com as fontes raster, ela
    deixava 2.008 patches de floresta e **1 de pastagem** no treino do Tapajós. Patches de pastagem de 2,24 km
    quase sempre têm remanescentes de floresta sendo abertos: a mediana é 8% do patch em 2019–2025 no
    Hansen, no MapBiomas e no DETER. Ou seja, "pastagem estável" quase não existe nessa escala. A regra
    por janela dá 11.360 janelas de floresta e 544 de pastagem (278 patches) no treino do Tapajós.
    As colunas `normal_treino` / `normal_estrito` continuam em `grade.csv` só para referência.

## 3. Divisão dos dados

- **Espacial:** blocos de 4 × 4 patches (8,96 km) inteiros em train/val/test (70/15/15, guloso por número de patches, semente 42). Nenhum bloco aparece em duas partições (verificado por `assert`).
- **Temporal:** treino só com sequências que terminam até 2022Q4 (`periodo = 'treino'`). O teste usa eventos de 2023–2025 (`periodo = 'teste'`).
- **Treino do modelo:** amostras normais (por janela) de `train`, com T ≤ 2022. **Ajuste de hiperparâmetros
  e limiar:** amostras normais e patch-passos limpos de `val`, nunca rótulos de anomalia.
- **Teste:** blocos `test` do Tapajós em 2023–2025; e `norte_ro` inteiro (`test_gen`) em 2023–2025.

- **Instantâneo dos rótulos:** o INPE edita PRODES/DETER depois de publicar. Entre 15/10 e 29/10, o PRODES
  do Tapajós passou de 8.507 para 8.506 polígonos. Os GeoPackages em `rotulos/<aoi>/cache/` no Drive
  (primeiro download) são a referência; não apagar.
- **Sub-AOI de 16 × 16 patches (Etapa 3):** foi escolhida por ter muitos eventos e serve só para
  desenvolvimento e inspeção. Métricas reportadas usam as AOIs completas, nunca só a sub-AOI, porque a
  escolha olhou os rótulos e infla a prevalência.

## 4. Métricas (`avaliacao.py`)

- **Prevalência real:** positivos = eventos; negativos = todos os patch-passos limpos. Sem rebalancear.
- **Por tarefa:** `todos` e cada tipo. Os negativos de um tipo excluem eventos de outros tipos.
- **AUROC e AUPRC** com IC 95% por bootstrap de blocos espaciais (300 reamostragens). **TPR a 1% e 5% de FPR**, **precisão@100 e @500**.
- **Limiar operacional:** quantil dos scores limpos da validação para FPR = 1% (`threshold_at_fpr`).
- **Por evento:** fração dos polígonos PRODES/DETER detectados (algum patch tocado passa do limiar entre o trimestre do evento e 2 trimestres depois), por tipo e tamanho (0–5, 5–25, 25–100, >100 ha).
- **Antecedência vs. DETER:** primeiro trimestre (−4 a +4) em que um patch com ≥ 1% do alerta passa do limiar. Antes do alerta só contam trimestres limpos do patch. Sempre reportar junto com `chance_rate`: um aviso "antecipado" ao acaso é comum quando um alerta toca vários patches.

## 5. Números com todas as fontes (execução local de 29/10, `results/etapa2_rotulos/20261029-150113`)

| | Tapajós | Norte RO |
|---|---|---|
| Patch-anos com evento, teste 2023–2025 | 23–29% (+27% ambíguos) | 29% (+40% ambíguos) |
| Patch-trimestres com evento (DETER + MCD64) | 4–5% | 3–6% |
| Normais pela regra antiga (por patch) | 2.876 (só 1 de pastagem no train) | 2.037 |

Com Hansen e MapBiomas, a prevalência anual por patch fica alta: um patch de 2,24 km com ≥ 1% de perda
em algum ano é comum na fronteira. Por isso a avaliação **trimestral** (data do DETER) é a mais
exigente e a mais próxima do uso prático. A anual serve para comparar com o AlphaEarth.

### Primeira execução (15/10, só PRODES + DETER)

| | Tapajós | Norte RO |
|---|---|---|
| Polígonos PRODES / DETER (2019–2025) | 8.507 / 13.104 | 11.319 / 14.488 |
| Patch-anos com evento (treino → teste) | 16% → 12–16% | 21% → 11% |
| Patch-trimestres com evento | 3–4% | 3–5% |
| Patches normais de treino | 50% | 52% |

Concordância de desmatamento por patch-ano: 73% dos patch-anos com DETER também têm PRODES
(76% em RO). Só 37% dos patch-anos com PRODES têm DETER (60% em RO), porque o DETER não pega
cortes pequenos. As fontes raster devem reduzir os normais de treino. Conferir a tabela de normais
por cobertura depois da execução no Colab.
