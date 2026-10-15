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
- **Normal de treino:** em **cada** fonte, a fração somada em 2019–2025 fica abaixo de 0,1% do patch (~0,5 ha).
  A tolerância absorve pixels isolados de ruído de MapBiomas/Hansen. A coluna `normal_estrito` (zero absoluto) fica salva para comparação.
  Normais incluem toda cobertura estável (floresta, pastagem, agricultura, água).

## 3. Divisão dos dados

- **Espacial:** blocos de 4 × 4 patches (8,96 km) inteiros em train/val/test (70/15/15, guloso por número de patches, semente 42). Nenhum bloco aparece em duas partições (verificado por `assert`).
- **Temporal:** treino só com sequências que terminam até 2022Q4 (`periodo = 'treino'`). O teste usa eventos de 2023–2025 (`periodo = 'teste'`).
- **Treino do modelo:** patches `normal_treino` de `train`, até 2022. **Ajuste de hiperparâmetros e limiar:** patches `val` (limpos), nunca rótulos de anomalia.
- **Teste:** blocos `test` do Tapajós em 2023–2025; e `norte_ro` inteiro (`test_gen`) em 2023–2025.

## 4. Métricas (`avaliacao.py`)

- **Prevalência real:** positivos = eventos; negativos = todos os patch-passos limpos. Sem rebalancear.
- **Por tarefa:** `todos` e cada tipo. Os negativos de um tipo excluem eventos de outros tipos.
- **AUROC e AUPRC** com IC 95% por bootstrap de blocos espaciais (300 reamostragens). **TPR a 1% e 5% de FPR**, **precisão@100 e @500**.
- **Limiar operacional:** quantil dos scores limpos da validação para FPR = 1% (`threshold_at_fpr`).
- **Por evento:** fração dos polígonos PRODES/DETER detectados (algum patch tocado passa do limiar entre o trimestre do evento e 2 trimestres depois), por tipo e tamanho (0–5, 5–25, 25–100, >100 ha).
- **Antecedência vs. DETER:** primeiro trimestre (−4 a +4) em que um patch com ≥ 1% do alerta passa do limiar. Antes do alerta só contam trimestres limpos do patch. Sempre reportar junto com `chance_rate`: um aviso "antecipado" ao acaso é comum quando um alerta toca vários patches.

## 5. Números da primeira execução (só PRODES + DETER; as fontes raster rodam no Colab)

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
