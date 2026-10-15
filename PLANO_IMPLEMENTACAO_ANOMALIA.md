# Plano de Implementação — Detecção de Anomalias na Amazônia

Este documento transforma o [PLAN.md](PLAN.md) em etapas executáveis. Ele parte do que já
existe no repositório e do resultado do *first pass* em
[experiments/amazon_anomaly_firstpass/](experiments/amazon_anomaly_firstpass/README.md).

**Prazo da entrega:** final de novembro de 2026 (~6 semanas a partir de 15/10).
**Alvo posterior:** artigo para o IGARSS 2027 (conferir no site oficial a data de submissão).

---

## 0. Ponto de partida: o que o first pass já mostrou

No first pass usamos embeddings AlphaEarth (no lugar do TerraMind) e rótulos MapBiomas
na região do Tapajós / BR-163, sem treinar nenhum modelo.

| Tarefa (blocos espaciais de teste) | Mudança no embedding (cosseno) | Melhor método espectral |
|---|---|---|
| Todas as anomalias vs. todos os normais | **0,89** AUROC | 0,82 (queda de NDVI) |
| Desmatamento vs. floresta estável | **0,99** | 0,93 |
| Desmatamento vs. pastagem estável | **0,85** | 0,76 |
| Mineração vs. normais | **0,87** | 0,82 |

**Conclusão:** um foundation model de EO separa perturbações reais de terreno normal melhor
que features espectrais. A hipótese central do PLAN.md se sustenta.

**Pontos de atenção que este plano precisa resolver:**

1. **Baseline estático forte.** Um kNN sobre o embedding do ano T, sem nenhuma informação
   temporal, obteve AUROC 0,93 (0,95 em mineração). O Temporal Transformer AE só se
   justifica se superar esse baseline. Isso vira um experimento obrigatório (Etapa 7).
2. **Rótulos não independentes.** MapBiomas e AlphaEarth derivam de dados de satélite
   parecidos. A avaliação final deve usar fontes independentes (PRODES, DETER etc.).
3. **Amostragem 50/50.** No mundo real, menos de 1% dos pixels sofre perturbação.
   Precisamos de métricas na prevalência real.
4. **Resolução anual.** O PLAN.md prevê séries X(t1…tn). Para isso precisamos de
   composições sub-anuais (sazonais ou mensais), que o AlphaEarth não oferece.

---

## Etapa 1 — Ambiente e infraestrutura

**Objetivo:** conseguir rodar PyTorch, TerraTorch e scikit-learn com GPU.

> **Status (15/10):** código em [notebooks/gabriel/](notebooks/gabriel/) (`config.py`, `utils.py`,
> `01_ambiente.ipynb`). Máquina local: Smart App Control ativo (bloqueia pandas, sklearn e partes do
> scipy), sem GPU NVIDIA e sem WSL. **Decisão: o pesado roda no Google Colab, com dados no Drive.**

- [x] **Desbloquear as bibliotecas.** Resolvido via Colab; local fica só com numpy/matplotlib. O Controle de Aplicativo do Windows (Smart App
      Control) bloqueia as DLLs de `torch` e `scipy` no `venv` atual. Opções, da mais
      simples para a mais robusta:
      - desativar o Smart App Control (não tem volta sem reinstalar o Windows; decidir antes);
      - usar WSL2 + Ubuntu com o mesmo código;
      - rodar treino e extração de embeddings no Google Colab / Kaggle (GPU gratuita) e
        manter a máquina local para análise e mapas.
- [x] **Definir onde roda o pesado.** Extração com TerraMind e treino do Transformer
      precisam de GPU. Recomendação: Colab ou Kaggle, com dados no Google Drive ou GCS.
- [x] **Organizar o código** (feito como notebooks + `config.py`/`utils.py` em `notebooks/gabriel/`,
      no lugar do pacote abaixo) em um pacote, separado dos notebooks da disciplina:
      ```
      src/anomalia/
        config.py          # AOIs, anos, sementes, caminhos (ou configs/*.yaml)
        dados/             # ingestão Sentinel, máscaras de nuvem, patches, séries
        rotulos/           # PRODES, DETER, MapBiomas, fogo -> rótulos de avaliação
        embeddings/        # TerraMind (e AlphaEarth como comparação)
        modelos/           # Transformer AE, Isolation Forest, Deep SVDD
        avaliacao/         # métricas, bootstrap por blocos, mapas
      scripts/             # 01_baixar_dados.py, 02_extrair_embeddings.py, 03_treinar.py ...
      configs/             # um YAML por experimento
      ```
      Reaproveitar `common.py` (estratos MapBiomas, composição S2 com Cloud Score+) e as
      métricas de `analyze.py` do first pass.
- [x] **Novo `requirements.txt`** com versões fixadas: `terratorch`, `torch`,
      `rasterio`, `xarray`, `zarr`, `scikit-learn`, `earthengine-api`, `geopandas`.
      *Feito: `requirements-colab.txt` + `requirements-colab.lock.txt` (versões do Colab).*

**Resultado (15/10, Colab T4):** `torch.cuda.is_available() == True`; TerraMind Small carrega como
`terratorch_terramind_v1_small` (terratorch 1.2.13) e devolve 12 camadas de 196 × 384 tokens em
~10 ms/patch. Log em `results/etapa1_ambiente/20261015-183140/`. O gate da Etapa 4 (TerraMind nos
pontos do first pass) continua aberto: o teste aqui usou um tensor aleatório.
- [x] **Sementes fixas e log de experimentos.** (`utils.set_seed` e `utils.new_run`) Cada execução salva sua config, sua
      semente e o commit do git junto aos resultados.

**Entregável:** `python -c "import terratorch, torch; print(torch.cuda.is_available())"` retorna `True` no ambiente de treino.

---

## Etapa 2 — Recorte, rótulos de referência e protocolo de avaliação

Definir isso **antes** de treinar evita ajustar o protocolo ao resultado.

> **Status (15/10):** protocolo fixado em [notebooks/gabriel/PROTOCOLO.md](notebooks/gabriel/PROTOCOLO.md);
> código em `rotulos.py`, `avaliacao.py` e `02_rotulos.ipynb`. PRODES e DETER já foram testados localmente
> nas duas AOIs. **Falta** rodar o notebook no Colab para as fontes do Earth Engine (Hansen, MapBiomas,
> MCD64A1) e conferir os normais por cobertura.

### 2.1 Áreas de estudo (AOIs)
- [x] **AOI principal:** Tapajós / BR-163 (PA), a mesma do first pass. Tem desmatamento,
      garimpo e queimadas.
- [x] **AOI de generalização:** uma segunda região nunca vista no treino. Sugestões:
      sul do Amazonas (Lábrea/Apuí) ou norte de Rondônia. Serve para responder "o modelo
      aprendeu o normal da floresta ou só desta região?".
- [x] **Período:** 2019–2025. O Sentinel-2 L2A tem cobertura confiável na Amazônia a
      partir do fim de 2018.

### 2.2 Rótulos (somente para avaliação; o modelo não os vê)

| Fonte | O que dá | Uso |
|---|---|---|
| **PRODES** (INPE) | Polígonos anuais de corte raso | Rótulo principal de desmatamento, independente do MapBiomas |
| **DETER** (INPE) | Alertas quase diários | Data do evento; medir antecedência da detecção |
| **MapBiomas Mineração / LULC** | Classe 30 e transições anuais | Garimpo; continuidade com o first pass |
| **MapBiomas Fogo / MODIS MCD64A1** | Cicatrizes de queimada | Fogo em floresta que continua floresta (anomalia "difícil") |
| **MapBiomas Água** | Superfície d'água anual | Alterações em rios e corpos d'água |
| **Hansen GFC** (`lossyear`) | Perda de cobertura arbórea | Checagem cruzada |

- [x] Script que gera, para cada patch e cada passo de tempo, o rótulo de cada tipo de
      perturbação, a fração do patch afetada e a data do evento.
- [x] Definir **"normal" para o treino**: patches sem nenhum evento em nenhuma fonte, no
      período inteiro de treino. Incluir floresta, pastagem, agricultura e água estáveis,
      porque o modelo precisa aprender que pastagem varia sem ser anomalia.

### 2.3 Divisão dos dados
- [x] **Espacial:** blocos de ~10 km; nenhum bloco aparece em mais de uma partição
      (mesma lógica de `assign_blocks_to_splits` já usada no repositório).
- [x] **Temporal:** treino com sequências terminando até 2022; avaliação em eventos de
      2023–2025.
- [x] **AOI de generalização:** só teste.

### 2.4 Métricas
- [x] **AUROC e AUPRC na prevalência real** (sem balancear as classes no teste).
- [x] **TPR a 1% e 5% de FPR** e **precisão@k**: das k regiões mais anômalas do mapa,
      quantas são eventos reais (é como o mapa seria usado na prática).
- [x] **Métricas por evento:** fração dos polígonos PRODES/DETER detectados, além da
      contagem por pixel.
- [x] **Antecedência:** quantos dias depois do alerta DETER (ou antes do PRODES) o modelo
      sinaliza o evento.
- [x] Intervalos de confiança com **bootstrap por blocos** (já implementado no first pass).
- [x] Tudo reportado **por tipo de perturbação**.

**Entregável:** `rotulos/` gerando um arquivo de rótulos por AOI e um documento curto
fixando o protocolo.

---

## Etapa 3 — Dados Sentinel e séries temporais

### 3.1 Aquisição
- [ ] **Sentinel-2 L2A** (bandas que o TerraMind espera para `S2L2A`; conferir a lista na
      documentação do TerraTorch) com máscara de nuvem Cloud Score+.
- [ ] **Sentinel-1 GRD/RTC** (VV, VH). Na Amazônia a nuvem é o maior problema e o radar
      enxerga através dela. O TerraMind aceita S1 nativamente, então vale incluir desde o
      início, mesmo que o primeiro experimento use só S2.
- [ ] **Composições por período:** começar com **trimestral** (4 passos/ano; ~28 passos
      em 2019–2025). Mediana dentro do período com nuvens mascaradas, e uma máscara de
      "dado faltante" quando não houver observação limpa (comum de dez. a abr.).
- [ ] **Exportação:** Earth Engine → GeoTIFF (Drive/GCS), ou STAC do Microsoft Planetary
      Computer. Estimar o volume antes de baixar.

### 3.2 Patches
- [ ] Grade regular de patches **224 × 224 px a 10 m** (~2,24 km), o tamanho de entrada
      do TerraMind.
- [ ] Guardar como **cubo** `(patch, tempo, canal, H, W)` em Zarr, com coordenadas,
      bloco espacial e máscara de nuvem por passo.
- [ ] Normalizar com as estatísticas que o TerraMind espera por modalidade (não com
      estatísticas próprias).

**Entregável:** cubo Zarr da AOI principal e um notebook de inspeção visual (alguns patches
ao longo do tempo, com e sem evento).

---

## Etapa 4 — Modelo 1: embeddings com TerraMind Small

- [ ] Carregar o **TerraMind 1.0 Small** pelo TerraTorch (backbone pré-treinado, congelado).
      Conferir o nome exato no registry da versão instalada.
- [ ] Para cada patch e cada passo de tempo, extrair:
      - **embedding do patch** `Z(t)`: média dos tokens de saída (um vetor por patch);
      - **grade de tokens** (14 × 14 tokens de 16 px ≈ 160 m): permite mapas mais finos
        que o patch inteiro.
- [ ] Passos sem dado (nuvem) recebem máscara, não zeros.
- [ ] Salvar os embeddings em disco (Zarr/npz). Todos os modelos seguintes partem deles,
      então a extração com GPU roda uma vez só.
- [ ] **Teste de sanidade (gate):** repetir o score de mudança por cosseno do first pass
      com TerraMind no lugar do AlphaEarth, nos mesmos pontos. Se o TerraMind ficar muito
      abaixo do AlphaEarth (ex.: AUROC < 0,80 em "todas as anomalias"), investigar
      normalização, bandas e composição antes de seguir.

**Entregável:** embeddings TerraMind de toda a AOI e tabela comparando TerraMind,
AlphaEarth e espectral no teste do first pass.

---

## Etapa 5 — Baselines

Todos treinados **só com sequências normais** e avaliados no mesmo protocolo.

| Baseline | Entrada | Pergunta que responde |
|---|---|---|
| Features espectrais → Isolation Forest | Médias/desvios por banda + NDVI/NBR, e a diferença para o histórico | O FM ajuda em relação a features clássicas? |
| TerraMind → Isolation Forest | `Z(t) − média(Z(t−k…t−1))` | Um detector simples sobre o FM já basta? |
| TerraMind → Deep SVDD | Idem | Um detector profundo de uma classe ajuda? |
| **TerraMind estático → kNN** | Só `Z(t)`, sem histórico | **O tempo importa?** (o alerta do first pass) |
| Mudança por cosseno | `Z(t)` vs. média do histórico | Baseline sem parâmetros |
| AlphaEarth (anual) | Mesmos scores | Comparação com outro FM |

- [ ] Implementar com scikit-learn (Isolation Forest) e PyTorch (Deep SVDD).
- [ ] Ajustar hiperparâmetros **apenas** no conjunto de validação de normais (sem usar
      rótulos de anomalia).

---

## Etapa 6 — Modelo 2: Temporal Transformer Autoencoder

### 6.1 Arquitetura (ponto de partida)
- Entrada: sequência `Z(t−k), …, Z(t)` com k ≈ 8 (dois anos de trimestres).
- Projeção linear para d = 256, mais **codificação temporal** (ano e época do ano, em
  seno/cosseno), porque a sazonalidade é parte do "normal".
- Encoder: 4 camadas, 4 cabeças; **máscara de atenção** para passos sem dado.
- Gargalo de dimensão menor (ex.: 64) e decoder simétrico.
- Saída: sequência reconstruída `Ẑ`.

### 6.2 Cuidado: o autoencoder pode "copiar" a anomalia
Um autoencoder puro pode aprender a reconstruir qualquer coisa, inclusive o evento
anômalo. Testar as duas formulações e manter a melhor:

- **Reconstrução com máscara:** mascarar o último passo (ou passos aleatórios) e
  reconstruí-lo a partir do contexto.
- **Previsão:** prever `Z(t)` a partir de `Z(t−k…t−1)`.

Em ambas o modelo nunca vê `Z(t)` ao produzir `Ẑ(t)`, então um desvio do padrão aprendido
gera erro alto.

### 6.3 Treino
- [ ] Só sequências normais (Etapa 2.1), com perda MSE ou cosseno no espaço de embedding.
- [ ] Adam, early stopping pelo erro de reconstrução na validação de normais.
- [ ] Registrar hiperparâmetros e curva de perda.

### 6.4 Anomaly Score
- [ ] `score(t) = distância(Z(t), Ẑ(t))` (cosseno ou L2).
- [ ] **Normalização local:** converter em z-score relativo à distribuição de erro dos
      patches normais do mesmo tipo de cobertura ou região. Pastagem tem erro "normal"
      maior que floresta, e sem isso o mapa vira um mapa de pastagem.
- [ ] Variante por token (grade 14 × 14) para mapas de maior resolução.

**Entregável:** modelo treinado, checkpoint versionado e scores de todos os patches/tempos.

---

## Etapa 7 — Experimentos e ablações

| # | Experimento | Pergunta |
|---|---|---|
| E1 | Transformer AE vs. todos os baselines | Resultado principal do artigo |
| E2 | **Temporal vs. estático** (kNN em `Z(t)`) | O componente temporal agrega valor? |
| E3 | Espectral vs. TerraMind, mesmo detector | O FM melhora a detecção? (pergunta do PLAN.md) |
| E4 | Tamanho da janela k (2, 4, 8, 12) | Quanto histórico é necessário? |
| E5 | S2 apenas vs. S2 + S1 | O radar compensa as nuvens? |
| E6 | TerraMind Small vs. Base (se houver tempo/GPU) | O tamanho do modelo importa? |
| E7 | Treino na AOI principal, teste na AOI de generalização | O modelo generaliza? |
| E8 | Resultados por tipo de perturbação | Onde funciona e onde falha (fogo, água, estradas) |
| E9 | Antecedência vs. DETER | Detecta antes ou depois dos sistemas oficiais? |
| E10 | Reconstrução mascarada vs. previsão | Qual formulação do AE (Etapa 6.2) funciona melhor? |

- [ ] Cada experimento é uma config YAML e gera uma pasta `results/<experimento>/` com
      métricas, config e figuras.
- [ ] Tabela consolidada com IC 95% por bootstrap de blocos.

---

## Etapa 8 — Mapas de anomalia

- [ ] Montar os scores por patch e por token em um **raster georreferenciado** (GeoTIFF)
      por passo de tempo.
- [ ] Figuras: mapa de anomalia × polígonos PRODES/DETER × RGB Sentinel-2, no formato de
      [map_2022.png](experiments/amazon_anomaly_firstpass/results/map_2022.png).
- [ ] Limiar operacional: escolhido na validação para um FPR fixo (ex.: 1%), nunca no teste.
- [ ] **Estudos de caso:** 3 a 5 eventos conhecidos (um garimpo, um desmatamento grande,
      uma queimada, uma estrada nova), mostrando a série do score ao longo do tempo.
- [ ] **Análise de erros:** os falsos positivos mais fortes (o que são? nuvem residual,
      colheita, cheia de rio?) e os falsos negativos.

---

## Etapa 9 — Reprodutibilidade e entrega (final de novembro)

Checklist do PLAN.md:

- [ ] **Pipeline funcionando:** scripts numerados que rodam de ponta a ponta a partir das
      configs.
- [ ] **Modelos implementados e treinados:** Transformer AE + baselines, com checkpoints.
- [ ] **Experimentos executados:** pelo menos E1, E2, E3, E7 e E8.
- [ ] **Métricas:** tabela consolidada (Etapa 7).
- [ ] **Mapas de anomalias:** GeoTIFFs e figuras (Etapa 8).
- [ ] **Código reproduzível:** README com ordem de execução, versões, tempo aproximado de
      cada etapa e onde roda (local vs. Colab).
- [ ] **Resultados salvos:** `results/` com tudo necessário para refazer as figuras sem
      retreinar.

---

## Cronograma sugerido

| Semana | Período | Foco | Gate ao final |
|---|---|---|---|
| 1 | 15–21/10 | Etapa 1 (ambiente) + Etapa 2 (AOIs, rótulos, protocolo) | Torch com GPU funcionando; protocolo fixado |
| 2 | 22–28/10 | Etapa 3 (dados S2/S1, composições, patches) | Cubo Zarr da AOI principal |
| 3 | 29/10–04/11 | Etapa 4 (TerraMind) + início da Etapa 5 | **Gate:** TerraMind reproduz o first pass |
| 4 | 05–11/11 | Etapa 5 (baselines) + Etapa 6 (Transformer AE) | AE treinado; baselines avaliados |
| 5 | 12–18/11 | Etapa 7 (experimentos E1–E5, E7, E8) | Tabela principal |
| 6 | 19–30/11 | Etapa 8 (mapas, estudos de caso) + Etapa 9 (entrega) | **Entrega** |
| Férias | dez–jan | E6, E9, E10; redação do artigo IGARSS (4 páginas) | Submissão |

**Se o tempo apertar, cortar nesta ordem:** E6 (Base) → E9 (antecedência) → S1 (E5) →
AOI de generalização com menos pontos. **Não cortar** E2 (temporal vs. estático): é o
experimento que justifica o modelo principal.

---

## Riscos e mitigação

| Risco | Impacto | Mitigação |
|---|---|---|
| Bloqueio de DLLs no Windows persiste | Não treina localmente | Colab/Kaggle ou WSL2 desde a semana 1 |
| Nuvens deixam muitos passos vazios | Séries incompletas, AE instável | Máscara de atenção; composições trimestrais; incluir S1 |
| Transformer AE não supera o kNN estático | Contribuição temporal fraca | Ainda é um resultado publicável se bem documentado; testar formulação de previsão (6.2) e janelas maiores |
| AE reconstrói as próprias anomalias | Scores baixos para eventos reais | Reconstrução mascarada / previsão (6.2) |
| Volume de dados maior que o esperado | Estoura disco ou tempo de download | Começar por uma sub-AOI de ~50 × 50 km; estimar antes de exportar |
| Pastagem domina os falsos positivos | Mapa pouco útil | Normalização local do score (6.4) |
| Rótulos com erros ou atraso (PRODES é anual) | Métricas pessimistas | Combinar PRODES + DETER; métricas por evento; inspeção visual dos erros |

---

## Decisões em aberto (para alinhar entre a dupla)

1. ~~Onde rodar o pesado: Colab, Kaggle, WSL2 ou desativar o Smart App Control?~~
   **Decidido (15/10): Google Colab**, dados no Drive (`MyDrive/ProjetoGrassQuality`).
2. Composição trimestral ou mensal? (Mensal dá mais resolução temporal, mas com muito mais
   passos vazios.)
3. S1 entra já no primeiro modelo ou só como ablação (E5)?
4. ~~Qual a AOI de generalização?~~ **Decidido (15/10): norte de Rondônia** `[-64, -10, -62, -8]`.
5. Quem cuida de cada etapa (o README precisa da divisão de tarefas).
