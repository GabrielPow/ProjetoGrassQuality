# Detecção de perturbações na Amazônia sem rótulos

**IBM8924 — Projeto de Deep Learning · Grupo 1 · 2026.2**

Este projeto testa se uma IA consegue apontar, trimestre a trimestre, onde a floresta amazônica está sendo
perturbada (desmatamento, garimpo, fogo, degradação) **sem nunca ver um exemplo rotulado**. A ideia é aprender
como um lugar normal evolui e marcar o que foge disso.

O pipeline final junta duas implementações do grupo:

- **Implementação original** (`notebooks/01`–`07`): recortes Sentinel-2, índices espectrais (NDVI, GNDVI, BSI),
  textura por Fourier e um classificador de vigor de pastagem por regressão logística.
- **Implementação TerraMind** (`notebooks/gabriel/`): cubo trimestral Sentinel-2 + Sentinel-1 (2019–2025),
  embeddings do *foundation model* TerraMind (IBM + ESA) e um **Transformer autoencoder temporal** treinado só
  com lugares normais.

A junção (`08_fusao`, `09_ae_combinado`) leva as 20 features clássicas da implementação original para a mesma
grade de células de 320 m do TerraMind e testa dois jeitos de combiná-las com o autoencoder.

> **Aviso.** Os resultados abaixo vêm de uma **sub-área de teste** (35,8 km × 35,8 km, 256 patches, 1,3% do
> planejado). As áreas de estudo completas ainda não foram processadas. Os números são preliminares e têm
> intervalos de confiança largos (ver [Limitações](docs/RELATORIO_TECNICO.md#63-limitações)).

---

## 1. Pergunta de pesquisa e hipóteses

**Pergunta central.** Embeddings de um *foundation model* (TerraMind), analisados ao longo do tempo, detectam
perturbações na Amazônia sem rótulos melhor do que features espectrais clássicas e do que o mesmo embedding sem
histórico?

| | Hipótese | Resultado na sub-área |
|---|---|---|
| **H1** | Um score de anomalia sobre embeddings de *foundation model* separa perturbação de normal melhor que a queda de NDVI. | Confirmada no teste rápido com AlphaEarth (0,89 × 0,82). **Não confirmada** com TerraMind no teste limitado (AE 0,67 × NDVI 0,77, ICs sobrepostos). |
| **H2** | Olhar o histórico (comparar com trimestres anteriores) detecta melhor que olhar só o trimestre atual. | Ganho pequeno por trimestre (0,73 × 0,71). O Transformer AE ainda não supera (0,68). |
| **H3** | Juntar features clássicas e TerraMind detecta melhor que cada um sozinho. | Inconclusiva: ajudou no diagnóstico (0,68 → 0,72), piorou no teste limitado (0,63). |
| **H4** | Dar a nota por célula pequena (320 m) detecta melhor que por patch inteiro (2,24 km). | Confirmada: quanto maior o pedaço, pior; eventos pequenos somem na média. |

As regras de avaliação foram fixadas **antes** de qualquer resultado, em
[notebooks/gabriel/PROTOCOLO.md](notebooks/gabriel/PROTOCOLO.md).

---

## 2. Instalação e execução

### Requisitos

- Python 3.13 (testado em Windows 11, CPU i5-13420H, 16 GB RAM, **sem GPU**)
- Conta no [Google Earth Engine](https://earthengine.google.com) com um projeto GCP habilitado
- ~10 GB livres para o cubo da sub-área; ~0,5 GB para os embeddings das áreas completas

### Instalação

```bash
python -m venv venv
venv\Scripts\activate                 # Windows  (macOS/Linux: source venv/bin/activate)

# Pipeline principal (TerraMind + autoencoder), versões fixadas:
pip install -r notebooks/gabriel/requirements-local.txt

# Pipeline original de pastagem e app Streamlit:
pip install -r requirements.txt

earthengine authenticate               # uma vez
```

### Ordem de execução

**Pipeline principal (detecção de anomalias)**, em `notebooks/gabriel/`:

| Etapa | Notebook | O que faz |
|---|---|---|
| 1 | `01_ambiente` | Valida TerraMind, terratorch e Earth Engine |
| 2 | `02_rotulos` | Baixa DETER, PRODES, Hansen, MapBiomas, MODIS; grade de patches; divisão espaço-tempo |
| 3 | `03_cubo_sentinel` | Cubo trimestral S2 + S1 da sub-área (16 × 16 patches, 28 trimestres) |
| 4 | `04_inspecao_cubo` | Disponibilidade, nuvens, exemplos antes/depois |
| 5 | `05_embeddings` | TerraMind Small congelado → 384 números por célula de 320 m; features clássicas |
| 6 | `06_baselines` | Métodos simples: queda de NDVI, kNN, Isolation Forest, Deep SVDD |
| 7 | `07_transformer_ae` | Treina o Transformer AE (masked × forecast) só com normais |
| 8 | `08_fusao` | Fusão de scores AE + detectores nas features clássicas |
| 9 | `09_ae_combinado` | AE com entrada combinada (TerraMind + 20 features) |

Tempo total da sub-área: ~1 h em CPU, a maior parte no download do Earth Engine.

Depois, `notebooks/apresentacao.ipynb` gera as figuras e tabelas em `results/apresentacao/`. Os slides finais
estão em [notebooks/apresentacao_v2.slides.html](notebooks/apresentacao_v2.slides.html).

Para processar as **áreas completas** (Tapajós + Rondônia, ~19 mil patches, ~3 dias, retomável):
`python notebooks/gabriel/stream_embeddings.py`. Este passo ainda não foi executado.

**Pipeline original (vigor de pastagem, entregáveis E1–E4)**, em `notebooks/`:
`01_dados` → `02_caracterizacao` → `03_baseline`. Os notebooks `04`–`07` são explorações (conversão
pastagem → lavoura, exploração de mudança na Amazônia, Fourier) que alimentaram a junção.

### Reprodutibilidade

- Semente 42 em todos os sorteios e treinos.
- Cada execução grava configuração, commit e versões em `results/<etapa>/<data>/run.json`.
- Rótulos de evento são usados **só para avaliar**; nenhum hiperparâmetro ou limiar é escolhido com eles.

---

## 3. Estrutura do repositório

```
ProjetoGrassQuality/
├── README.md                     # este arquivo
├── LICENSE
├── docs/
│   ├── RELATORIO_TECNICO.md      # relatório técnico completo
│   ├── RELATORIO_EXECUTIVO.md    # resumo de 1–2 páginas para não técnicos
│   └── historico/                # READMEs anteriores (trilha de pastagem / AlphaEarth)
├── notebooks/
│   ├── 01_dados … 03_baseline    # implementação original: vigor de pastagem (Goiânia)
│   ├── 04 … 07                   # explorações: conversão, mudança na Amazônia, Fourier
│   ├── dataset_utils.py          # NDVI, GNDVI, features agregadas
│   ├── apresentacao.ipynb        # gera figuras/tabelas da apresentação
│   ├── apresentacao_v2.slides.html
│   └── gabriel/                  # implementação TerraMind + junção
│       ├── 01 … 09 *.ipynb       # pipeline em etapas (tabela acima)
│       ├── PROTOCOLO.md          # regras de avaliação fixadas antes dos resultados
│       ├── config.py             # áreas, período, grade, parâmetros
│       ├── rotulos.py            # rótulos de evento (só avaliação) e divisão
│       ├── dados.py              # download do cubo Sentinel via Earth Engine
│       ├── embeddings.py         # TerraMind → fichas por célula
│       ├── features_originais.py # 20 features clássicas por célula (NDVI, BSI, Fourier…)
│       ├── baselines.py          # NDVI, kNN, IForest, Deep SVDD
│       ├── modelo.py             # Transformer autoencoder temporal
│       ├── fusao.py              # fusão de scores (z-score robusto)
│       ├── avaliacao.py          # AUROC, AUPRC, TPR@FPR, bootstrap por blocos
│       └── stream_embeddings.py  # processa as áreas completas sem guardar pixels
├── experiments/amazon_anomaly_firstpass/   # teste rápido com AlphaEarth (P0)
├── model/ + app.py               # modelo bônus: AlphaEarth + LSTM para pastagem, app Streamlit
├── utils/                        # download Earth Engine, regiões
├── data/processed.zip            # dados tratados da trilha de pastagem (E1)
├── results/                      # métricas, figuras e run.json de cada etapa
├── PLANO_IMPLEMENTACAO_ANOMALIA.md
└── PLANO_JUNTAR_DEEP_LEARNING.md # plano da junção das duas implementações
```

---

## 4. Resumo dos resultados

**Teste limitado**: blocos de validação da sub-área do Tapajós em 2023–2025, nunca vistos nesse período
(29 patch-trimestres com evento, 537 limpos, prevalência real ~5%). Nota do patch = maior nota das 49 células.

| Método | AUROC | IC 95% | AUPRC | TPR @ 1% FPR | Desmat. (8) | Fogo (17) | Garimpo (9) |
|---|---|---|---|---|---|---|---|
| Queda de NDVI (sem IA) | **0,77** | 0,63–0,82 | 0,16 | 0,00 | 0,73 | **0,90** | **0,65** |
| kNN no delta (TerraMind) | 0,74 | 0,63–0,82 | **0,25** | **0,07** | 0,78 | 0,89 | 0,53 |
| Deep SVDD (features clássicas) | 0,69 | 0,48–0,72 | 0,11 | 0,00 | **0,80** | 0,78 | 0,54 |
| Fusão de scores (AE + IForest clássico) | 0,68 | 0,61–0,76 | 0,08 | 0,00 | 0,75 | 0,77 | 0,53 |
| Transformer AE (TerraMind) | 0,67 | 0,58–0,74 | 0,09 | 0,00 | 0,75 | 0,77 | 0,49 |
| Transformer AE (features clássicas) | 0,66 | 0,53–0,80 | 0,08 | 0,03 | 0,68 | 0,73 | 0,60 |
| Transformer AE (TerraMind + clássicas) | 0,63 | 0,52–0,77 | 0,08 | 0,03 | 0,66 | 0,71 | 0,57 |

AUPRC ao acaso ≈ 0,05. Colunas por tipo de evento: AUROC.

**Em uma frase:** na sub-área, a IA não superou os métodos simples. O Transformer AE ordena eventos melhor que o
acaso (0,63–0,67), mas no limiar de 1% de alarme falso não marca nenhum dos 29 eventos. As diferenças cabem na
incerteza de uma amostra tão pequena; o teste nas áreas completas decide a pergunta central.

**Trilha de pastagem (entregáveis E1–E4):** regressão logística sobre médias/desvios por banda + NDVI/GNDVI em
1.800 recortes Sentinel-2 64 × 64 × 6 → acurácia **0,64** e F1 macro **0,63** no teste, contra 0,25 e 0,13 do
baseline trivial.

---

## 5. Relatórios

- **[Relatório técnico completo](docs/RELATORIO_TECNICO.md)**: introdução, revisão bibliográfica, metodologia,
  resultados, discussão e referências.
- **[Relatório executivo](docs/RELATORIO_EXECUTIVO.md)**: 1–2 páginas para quem não é da área técnica.
- [Protocolo de avaliação](notebooks/gabriel/PROTOCOLO.md) ·
  [Plano da junção](PLANO_JUNTAR_DEEP_LEARNING.md) ·
  [Slides](notebooks/apresentacao_v2.slides.html)

---

## 6. Autores, divisão de tarefas e uso de IA

| Integrante | Responsabilidades principais |
|---|---|
| **Gabriel Pow** | Implementação TerraMind (`notebooks/gabriel/`): rótulos, cubo Sentinel, embeddings, baselines de anomalia, Transformer AE, protocolo; junção das implementações; apresentação |
| **Pedro Pintor** | Implementação original (`notebooks/01`–`07`): ingestão Sentinel-2, caracterização, baseline de pastagem, explorações de mudança e Fourier; utilitários Earth Engine (`utils/`) |

**IA generativa.** Claude Code (Anthropic) foi usado para explorar o código, juntar as duas implementações,
escrever código, rascunhar textos (incluindo este README e os relatórios) e gerar figuras. As decisões de
protocolo e os resultados foram revisados pela dupla.

## 7. Licença

Código sob a licença [MIT](LICENSE). Os dados de terceiros seguem as licenças das fontes: Sentinel-1/2
(Copernicus, uso livre), MapBiomas (CC BY-SA 4.0), Hansen GFC (CC BY 4.0), DETER/PRODES (INPE, acesso
aberto), MODIS MCD64A1 (NASA, uso livre), TerraMind (pesos abertos, Apache 2.0).
