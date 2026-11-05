# Plano — Juntar as duas implementações

Escrito em 04/11/2026. Complementa o [PLANO_IMPLEMENTACAO_ANOMALIA.md](PLANO_IMPLEMENTACAO_ANOMALIA.md) e segue o
protocolo de [notebooks/gabriel/PROTOCOLO.md](notebooks/gabriel/PROTOCOLO.md). Nada aqui muda os rótulos, a divisão
dos dados ou as métricas.

**Objetivo:** usar a engenharia de features da implementação original (`notebooks/`) junto com o modelo treinado
na implementação do Gabriel (`notebooks/gabriel/`, Temporal Transformer AE sobre embeddings TerraMind), e medir
se a combinação detecta perturbações na Amazônia melhor que cada parte sozinha. O passo final (Fase 6) é um notebook
de slides para a apresentação, com a estrutura da implementação, exemplos de imagem, o porquê das ferramentas, o
teste final do modelo e as limitações.

---

## 1. O que cada implementação traz

| | Original (`notebooks/`) | Gabriel (`notebooks/gabriel/`) |
|---|---|---|
| Tarefa | 01–03: vigor de pastagem (Goiânia). 04: conversão pastagem → lavoura. 05–07: exploração de mudança na Amazônia | Detecção de anomalias não supervisionada na Amazônia (Tapajós, teste em Rondônia) |
| Dado | Recortes S2 de 64 × 64 px (6 bandas, 1 composição por ponto); 12 recortes 256 × 256 anuais | Cubo trimestral 2019–2025, patches 224 × 224 px, 12 bandas S2 + S1 (`data/cubo_tapajos_sub.zarr`) |
| Features | Média e desvio por banda, NDVI (média, desvio), GNDVI ([dataset_utils.py](notebooks/dataset_utils.py)); BSI, \|ΔNDVI\| e energia de alta frequência de Fourier em blocos ([07_fourier_mudanca.ipynb](notebooks/07_fourier_mudanca.ipynb)); relevo, chuva, vizinhança ([04_conversion_labels.ipynb](notebooks/04_conversion_labels.ipynb)) | TerraMind Small congelado: 384 números por célula de 320 m (49 células por patch); médias das 12 bandas + NDVI + NBR por célula |
| Modelo | Regressão logística (supervisionada) | Transformer AE (`modelo.py`), checkpoint em `checkpoints/transformer_ae_masked_tapajos_sub.pt`; baselines kNN, IForest, Deep SVDD (`baselines.py`) |

**Por que juntar.** No diagnóstico da sub-AOI (AUROC trimestral, `results/etapa5_*` e `results/etapa6_*`), os dois
tipos de sinal acertam coisas diferentes:

| Método | Todos | Desmatamento | Fogo | Degradação |
|---|---|---|---|---|
| Transformer AE (masked) | 0,68 | **0,81** | 0,68 | 0,70 |
| Queda de NDVI (espectral) | 0,71 | 0,76 | **0,76** | **0,76** |

O AE é melhor em desmatamento; o sinal espectral simples é melhor em fogo e degradação. Isso sugere que uma
combinação pode superar as duas partes. Esses números são diagnóstico (avaliação no split `train` da sub-AOI) e
**não devem ser reportados**.

---

## 2. Decisões: o que dá e o que não dá para juntar

| Caminho | Como | Usa o checkpoint atual? | Decisão |
|---|---|---|---|
| **A. Fusão de scores** | AE continua sobre TerraMind; features originais viram um segundo detector; os scores são combinados | Sim | **Principal.** Barato e não mexe no modelo |
| **B. AE com entrada combinada** | Entrada do AE = 384 (TerraMind) + ~20 (features originais) por célula; retreinar | Não (mesma arquitetura e código, pesos novos) | **Segundo passo**, se A ajudar |
| C. Classificador supervisionado em cima | Regressão logística sobre score do AE + features, treinada com rótulos de anomalia | — | **Não fazer.** O protocolo só permite rótulos na avaliação |
| D. AE na tarefa de pastagem (Goiânia) | Rodar o AE nos recortes de Goiânia | — | **Não fazer.** Não há série temporal, os recortes têm 64 px e 6 bandas (o TerraMind espera 224 px e 12), e o AE só viu floresta amazônica |

O checkpoint atual tem a entrada fixa em 384 dimensões (`Linear(384 → 128)`), por isso as features originais não
podem entrar nele diretamente. No caminho B o modelo é retreinado com o mesmo código.

---

## 3. Fase 0 — Registrar as regras antes de rodar

Para não escolher a combinação olhando o teste, registrar no `PROTOCOLO.md`, **antes de qualquer resultado**:

- [x] Lista fixa das features originais (seção 4).
- [x] Regra de fusão principal: **média dos z-scores robustos** (seção 6). Variantes (média e máximo dos
      percentis, outros pares) entram só como ablação.
- [ ] Pesos de perda do caminho B: 50% para o grupo TerraMind e 50% para o grupo de features originais.
- [x] Combinação principal a reportar: **AE (masked) + IForest nas features originais com delta**. As outras são
      ablação.
- [x] Nenhum hiperparâmetro é escolhido com rótulos de anomalia; só com normais de validação.

> **Feito em 04/11:** seção 6 do `PROTOCOLO.md`, escrita antes de qualquer resultado de fusão.

---

## 4. Fase 1 — Features originais por célula de 320 m

Novo módulo `notebooks/gabriel/features_originais.py`. Entrada: S2 de um patch `(12, 224, 224)` + `n_obs`.
Saída: `(49, F)` por patch e trimestre, na mesma grade de células (32 × 32 px) do TerraMind.

| Feature | Origem | Já existe no Gabriel? |
|---|---|---|
| Média de B2, B3, B4, B8, B11, B12 | `dataset_utils.aggregate_features` | Sim (`spec_cells`) |
| NDVI média | `dataset_utils` | Sim (`spec_cells`) |
| Desvio de B2, B3, B4, B8, B11, B12 (textura) | `dataset_utils` | **Não** (só no patch, não na célula) |
| NDVI desvio | `dataset_utils` | **Não** na célula |
| GNDVI média | `dataset_utils` | **Não** |
| BSI média (solo exposto) | `07_fourier_mudanca` | **Não** |
| Fourier do NDVI: energia em 4 anéis de frequência, com janela de Hann | `07_fourier_mudanca` (Teste 2) | **Não** |

Total: 7 já existentes + 13 novas = **20 features por célula**.

Detalhes:
- Reflectância = DN / 10000, como em `dataset_utils` e `embeddings.cell_spectral`.
- Pixels sem observação ficam fora das médias e desvios (máscara `n_obs > 0`, igual a `cell_spectral`). Célula com
  menos de 50% de pixels limpos vira NaN.
- **Fourier.** O Teste 2 do notebook 07 comparava blocos de 64 × 64 entre dois anos. Aqui cada célula (32 × 32,
  com Hann, média da célula removida) guarda o **perfil radial**: log10 da energia média em 4 anéis,
  (0; 0,04], (0,04; 0,08], (0,08; 0,125] e > 0,125 ciclo/pixel (o anel externo usa o mesmo corte do 07). A comparação com o passado é feita depois, como diferença entre o
  perfil em `t` e a média dos perfis em `t−4 … t−1`. Pixels faltantes são preenchidos com a média da célula antes da
  FFT. Guardar o perfil (4 números) em vez do espectro inteiro mantém o arquivo pequeno.
- Armazenar em `embeddings/<nome>_cells.zarr`, novo array `orig_cells` `(S, T, 49, 13)` float16, com os nomes
  em `attrs`. O `baselines._batches` junta `spec_cells` + `orig_cells` na leitura.

Tarefas:
- [x] `features_originais.py` com `cell_features_originais(s2, nobs) -> (N, 49, 13)` e testes simples: patch
      constante → desvio 0 e Fourier no piso; desvios, NDVI e GNDVI iguais aos de `dataset_utils` numa célula
      real; ordem das células igual à de `spec_cells`.
- [x] Sub-AOI: função `orig_from_cube(cube_path, out_path)` que lê o cubo e grava `orig_cells` (sem rodar o
      TerraMind; minutos em CPU). Feito: `(256, 28, 49, 13)`, 2% das células sem dado.
- [x] **AOIs completas (urgente):** chamar a nova função dentro de `embeddings.write_step`, **antes** de rodar
      `stream_embeddings.py` nas AOIs completas. O fluxo descarta os pixels; se as features não forem
      calculadas na passada, será preciso baixar tudo de novo do Earth Engine. Custo extra: desprezível em CPU,
      ~0,7 GB em disco para as duas AOIs.

---

## 5. Fase 2 — Detectores só com as features originais

Responde ao **E3** do plano principal (espectral × TerraMind, mesmo detector) com o conjunto de features completo
da implementação original, em vez das 14 médias atuais.

- [x] Em `baselines.py`, novo grupo **"Original"** com os mesmos detectores e a mesma entrada `[F, F − histórico]`:
  - IForest (300 árvores), kNN (k = 10), Deep SVDD;
  - \|ΔNDVI\| vs. histórico (já existe como "queda de NDVI"; manter);
  - energia de Fourier: soma de `(perfil(t) − perfil_histórico)²` nos anéis de alta frequência (o Teste 2 do 07,
    agora no tempo).
- [x] Ajuste só com células normais de treino (`treino_normal`), como hoje.
- [x] Conferir na sub-AOI que nada muda nos baselines antigos (kNN no delta TerraMind: 0,733 antes e depois; feito
      dentro do `08_fusao`, sem rerodar o `06`).

---

## 6. Fase 3 — Caminho A: fusão de scores com o AE treinado

**Regra de fusão (fixada a priori).** Para cada método `m`, padronizar o score em relação às
células/patches **normais de validação** (`val_normal`): `z_m(x) = (x − mediana) / IQR`. Assim todos ficam na
mesma escala, sem rótulos.

- Principal: **média dos z-scores robustos** dos métodos combinados.
- Ablação: **média** e **máximo dos percentis** (`p_m(x)` = fração dos normais de validação abaixo de x).
- Por que não percentis como principal: o percentil satura em 1 acima do maior normal, e as amostras mais
  anômalas empatariam entre si.

Duas versões, da mais barata para a mais fiel:

- [x] **J1 — fusão no patch.** Usa os scores de patch já salvos (`results/etapa6_*/scores_trimestrais.csv.gz` e
      `results/etapa5_*/scores_trimestrais.csv.gz`). Nada a recalcular; roda em segundos. Serve para ver se vale
      continuar.
- [x] **J2 — fusão na célula.** Combinar os percentis em cada uma das 49 células e só então tirar o máximo do
      patch. É mais fiel à ideia (o evento precisa aparecer na mesma célula nos dois sinais). Precisa guardar os
      erros por célula do AE (`modelo.score_all` já os calcula; hoje só o agregado é salvo). Carregar o
      checkpoint, sem retreinar: alguns minutos na sub-AOI.

Pares a avaliar (o principal está em negrito):

| Par | Motivo |
|---|---|
| **AE + IForest nas features originais** | Combinação principal |
| AE + queda de NDVI | Par mais simples; complementaridade vista no diagnóstico |
| AE + energia de Fourier | Fourier detectou bem mineração no notebook 07 |
| AE + kNN no delta TerraMind | Controle: duas visões do mesmo embedding |

- [x] Novo notebook `notebooks/gabriel/08_fusao.ipynb` com J1 e J2, saída em `results/etapa_fusao_<fonte>/`.

> **Status (04/11), Fases 1–3 na sub-AOI** (diagnóstico no split `train`, **não reportar**;
> `results/etapa_fusao_tapajos_sub/20261104-204250`; AUROC trimestral, todas as anomalias, 166 positivos):
> - **Combinação principal** (J2, AE + Original IForest, média dos z): 0,707 × 0,684 do AE (+0,023, IC pareado
>   −0,001 a +0,054). Ganho em **mineração** (+0,07, IC 0,04–0,10) e **degradação** (+0,04, IC 0,005–0,12);
>   desmatamento igual; AUPRC sem ganho.
> - A fusão **não supera** os melhores métodos sozinhos: kNN no delta TerraMind 0,733 e **Original Deep SVDD
>   0,737** (melhor resultado trimestral até agora; desmatamento 0,886). Contra a queda de NDVI, perde em fogo.
> - **Fourier** é o melhor método em mineração (0,789), como no notebook 07.
> - J1 ≈ J2. O máximo dos percentis zera o TPR@1% FPR (empates em 1), o que confirma a escolha do z robusto.
> - Referência da fusão: só 116 amostras normais de validação, todas floresta. O teste de verdade é a Fase 5.
> - **Sugestão:** incluir o AE-ORIG (Fase 4) mesmo sem ganho grande da fusão, porque as features originais com um
>   detector profundo (Deep SVDD) foram o melhor sinal.

---

## 7. Fase 4 — Caminho B: AE com entrada combinada

Só se J1/J2 mostrarem ganho, ou para fechar o E3.

- [x] `modelo.py`: parâmetro `entrada` (`'tm'`, `'orig'`, `'tm+orig'`) em `sequences`, `gather_training`,
      `Trainer` e `build_model(d_in=...)`. A validade passou a ser por célula: um passo com feature original NaN
      vira passo sem dado (máscara); no passo t, a célula fica sem score.
- [x] Padronização: por dimensão com os normais de treino (`nanmean`/`nanstd`).
- [x] Perda por grupo, com pesos fixos: `0,5 · MSE(TerraMind) + 0,5 · MSE(originais)` (`modelo.grupos`). O score
      segue a mesma fórmula. Registrado no `PROTOCOLO.md` (seção 6) antes do treino.
- [x] Treinar três modelos com a mesma arquitetura (d = 128, 4 camadas, masked), na mesma amostra:
  1. **AE-TM** (só TerraMind; é o checkpoint atual, para referência);
  2. **AE-ORIG** (só as 20 features originais): AE temporal sem foundation model (E3);
  3. **AE-TM+ORIG** (entrada combinada).
- [x] Custo: ~30 min por modelo na sub-AOI em CPU (medido no `07_transformer_ae`). Medido no `09`: AE-ORIG 10,5 min
      (24 épocas), AE-TM+ORIG 12,7 min (31 épocas), com parada antecipada.
- [x] Novo notebook `09_ae_combinado.ipynb`, salvando checkpoints com o nome da entrada
      (`transformer_ae_masked_<entrada>_<fonte>.pt`); reaproveita checkpoints existentes (`REFAZER = False`).

> **Checkpoints de treino (04/11):** `Trainer(..., resume_path=...)` grava a cada época modelo, otimizador,
> scheduler, estados aleatórios, curva e paciência (escrita atômica). Rodar o `09` de novo depois de um corte retoma
> da última época; teste: treino cortado na 3ª época e retomado termina com **pesos idênticos** ao treino sem corte.
> Configuração diferente = checkpoint ignorado. O parcial é apagado depois que o modelo final é salvo.
>
> **Status (04/11), Fase 4 na sub-AOI** (diagnóstico no split `train`, **não reportar**;
> `results/etapa_ae_combinado_tapajos_sub/20261104-211055`; AUROC trimestral, todas as anomalias):
> - **AE-ORIG 0,721** e **AE-TM+ORIG 0,720** × AE-TM 0,684 (+0,037 e +0,036; ICs pareados tocam o zero). Ganho
>   claro em **mineração** (+0,13 e +0,11, significativos); perda em desmatamento (−0,05, não significativa).
> - **J3 / E3:** com o mesmo AE, as features clássicas empatam ou superam o TerraMind.
> - **J4:** juntar na entrada (0,720) empata com juntar no score (fusão, 0,707; +0,013, IC −0,024 a +0,041).
> - AE-TM+ORIG ≈ AE-ORIG: dentro do AE, o embedding não acrescenta às features originais.
> - Os melhores continuam sendo detectores simples: Original Deep SVDD 0,737 e kNN no delta TerraMind 0,733.
> - AE-TM+ORIG começa a sobreajustar (404 dimensões, 26,6 mil sequências); a parada antecipada segurou.
> - **Para a Fase 5:** levar os três AEs, a fusão e o Deep SVDD original às AOIs completas. Conferências: o checkpoint antigo
> carregado pelo código novo reproduz os scores salvos do `07` (diferença 2e-7) e o mesmo conjunto de treino
> (26.656 sequências); as entradas `orig` e `tm+orig` treinam, salvam, recarregam e pontuam (0,6% das sequências
> de treino têm algum passo sem dado original). O `09` rodou de ponta a ponta numa cópia com 1 época (números sem
> valor, apagados). Falta rodar o `09` completo (~1 h em CPU).

---

## 8. Fase 5 — AOIs completas e teste

> **Adiada (05/11):** a apresentação técnica foi antecipada para 06/11, então a Fase 6 foi feita antes, com os números
> da sub-AOI. Estimativa para esta fase (a partir dos tempos medidos na sub-AOI): 60–80 h de download do Earth Engine (TerraMind em CPU roda junto,
> ~19 h) + ~12–16 h de baselines, treino e pontuação. Primeiro passo sugerido: `stream_embeddings.py tapajos
> --max-tasks 40` (~5 min) para medir a velocidade real e testar as features originais no fluxo.

- [ ] Rodar `stream_embeddings.py` em `tapajos` e `norte_ro` **com a Fase 1 já integrada**.
- [ ] Retreinar no Tapajós completo (AE-TM e, se for o caso, AE-TM+ORIG); ajustar detectores e percentis de fusão
      só com normais de `train` / `val`.
- [ ] Avaliar no split `test` do Tapajós e em `test_gen` (Rondônia), 2023–2025, com o `avaliacao.evaluate`.
- [ ] Testar a normalização local por cobertura (6.4), que não foi testável na sub-AOI.

---

## 9. Fase 6 — Notebook de slides para a apresentação

Um notebook Jupyter que vira apresentação (reveal.js) e mostra o trabalho inteiro: estrutura da implementação,
engenharia de features com imagens reais, o porquê de cada ferramenta, o teste final do modelo e as limitações.

Arquivo: `notebooks/apresentacao.ipynb` (fica em `notebooks/` porque usa as duas implementações; importa os
módulos de `notebooks/gabriel/` pelo `sys.path`).

### 9.1 Formato

- Cada célula recebe o tipo de slide nos metadados (`slideshow.slide_type`: `slide`, `subslide`, `fragment`,
  `skip`, `notes`). Texto do apresentador vai em células `notes`.
- Exportar sem o código, só com figuras e texto:
  `jupyter nbconvert notebooks/apresentacao.ipynb --to slides --no-input --SlidesExporter.reveal_scroll=True`
  → `notebooks/apresentacao.slides.html`, que abre no navegador.
- **Apresentar sem internet:** o nbconvert busca o reveal.js num CDN. Guardar uma cópia local e usar
  `--SlidesExporter.reveal_url_prefix=reveal.js`, ou levar também o PDF (imprimir o HTML com `?print-pdf`).
- **Roda só com o que está salvo:** lê o cubo, os embeddings, os checkpoints e `results/`. Não chama o Earth
  Engine nem treina nada. Meta: menos de 5 minutos do início ao fim. Cada figura também é salva em
  `results/apresentacao/` como PNG, para usar fora do notebook.
- Mesma paleta e estilo de figura dos notebooks `06_baselines` / `07_transformer_ae`.

### 9.2 Roteiro dos slides

| # | Slide | Conteúdo e figuras | Fonte dos dados |
|---|---|---|---|
| 1 | Título | Projeto, dupla, pergunta central: "um foundation model ajuda a detectar perturbações na Amazônia sem rótulos?" | — |
| 2 | Problema | Desmatamento, garimpo, fogo e degradação; por que aprender o "normal" e detectar o desconhecido; alvo IGARSS 2027 | `PLAN.md` |
| 3 | Duas implementações, um pipeline | Diagrama: original (features espectrais, Goiânia e exploração na Amazônia) + Gabriel (TerraMind, cubo trimestral, AE) → onde se juntam (Fases 1–4) | Este plano |
| 4 | Área e unidade de análise | Mapa do Tapajós e de Rondônia; grade de patches de 2,24 km, blocos de 8,96 km coloridos por split, célula de 320 m; 28 trimestres | `rotulos/<aoi>/grade.csv` |
| 5 | Como uma imagem se parece | Um patch com evento ao longo de 8 trimestres em RGB (B4, B3, B2); o mesmo patch em radar (VV); pixels sem observação (nuvem) em destaque; grade 7 × 7 de células por cima | `data/cubo_tapajos_sub.zarr` |
| 6 | Antes e depois de um evento | RGB, NDVI e BSI antes e depois de um desmatamento e de um garimpo, com o polígono DETER/PRODES por cima | cubo + rótulos |
| 7 | Rótulos (só avaliação) | Fontes (DETER, PRODES, Hansen, MapBiomas, MODIS); evento ≥ 1% do patch, ambíguo, normal por janela | `PROTOCOLO.md` |
| 8 | Engenharia de features (original) | Fórmulas com citação: NDVI (Rouse et al., 1974), GNDVI (Gitelson et al., 1996), BSI; desvio por banda como textura; resultado de origem: vigor de pastagem, regressão logística 0,64 de acurácia × 0,25 do trivial (`03_baseline`) | `dataset_utils.py`, notebooks 01–03 |
| 9 | Fourier como feature | Uma célula: NDVI, janela de Hann, espectro de magnitude, perfil nos 4 anéis antes × depois do evento; por que o perfil tolera desalinhamento e o que aprendemos no notebook 07 (Teste 1 falhou, Teste 2 funciona localmente) | cubo + `07_fourier_mudanca` |
| 10 | TerraMind | Imagem 224 × 224 → 196 tokens de 16 px → células de 2 × 2 tokens; mapa de mudança do embedding entre dois trimestres | `embeddings/tapajos_sub_tokens.zarr` |
| 11 | O modelo | Diagrama do Transformer AE: 9 trimestres, codificação de ano e trimestre, máscara para passos sem dado, gargalo de 64; masked × forecast; por que ele não "copia" a anomalia; curvas de perda | `modelo.py`, `results/etapa6_*/curva_*.csv` |
| 12 | Fusão | Score do AE + score das features originais → z-scores robustos nos normais de validação → média; regra fixada antes do teste | Fase 3 |
| 13 | Por que essas ferramentas | Tabela de decisões (9.3) | — |
| 14 | Protocolo | Divisão espacial por blocos e temporal (treino ≤ 2022, teste 2023–2025); prevalência real; por que AUROC/AUPRC e não acurácia | `PROTOCOLO.md` |
| 15 | **Teste final** | Ver 9.4 | `avaliacao.py` |
| 16 | Mapa de anomalias | Score por célula num trimestre × polígonos DETER × RGB; série do score de um evento conhecido ao longo do tempo | scores salvos + cubo |
| 17 | Onde erra | 2 falsos positivos fortes (nuvem residual? pastagem?) e 2 falsos negativos, com as imagens | scores + cubo |
| 18 | **Limitações** | Ver 9.5 | — |
| 19 | Trabalhos futuros | AOIs completas, generalização em Rondônia, TerraMind Base, antecedência vs. DETER, artigo IGARSS | Plano principal |
| 20 | Reprodutibilidade e uso de IA | Ordem de execução, onde roda, tempo; declaração de uso de IA generativa | README |

### 9.3 Por que escolhemos cada ferramenta (slide 13)

| Escolha | Alternativa | Motivo |
|---|---|---|
| Google Earth Engine | Baixar cenas (Copernicus, Planetary Computer) | Composições e máscara de nuvem (Cloud Score+) feitas no servidor; só baixamos o recorte final |
| Sentinel-2 + Sentinel-1 | Landsat; só óptico | Aberto, 10 m; o radar enxerga através das nuvens da Amazônia (S2 + S1 > só S2 em todos os métodos TerraMind) |
| Composição trimestral | Mensal, anual | Mensal deixa muitos passos vazios por nuvem; anual perde o momento do evento e não compara com o DETER |
| TerraMind Small (congelado) | AlphaEarth; treinar do zero | Pesos abertos: controlamos a entrada (bandas, período, recorte); AlphaEarth só existe anual. Sem rótulos e sem GPU, não dá para treinar um backbone |
| Célula de 320 m | Média do patch de 2,24 km | A média dilui eventos pequenos (gate: 0,57 no patch × 0,84 no token); o máximo das células venceu em 29 de 30 comparações |
| Features espectrais da implementação original | Só o embedding | Interpretáveis, baratas, e acertam fogo e degradação onde o AE é mais fraco |
| Transformer AE | LSTM (usado no protótipo de pastagem), autoencoder comum | A atenção lida com trimestres faltantes via máscara; a codificação de trimestre modela a sazonalidade; a máscara no passo t impede copiar a anomalia |
| Não supervisionado | Classificador com rótulos | Rótulos chegam atrasados e incompletos (DETER não vê cortes pequenos); queremos detectar tipos de evento não vistos |
| Zarr + processamento local em CPU | Google Drive / Colab | Drive com ~2 GB livres; Zarr grava por bloco e permite retomar |
| scikit-learn / PyTorch | — | IForest e kNN prontos; PyTorch para Deep SVDD e o AE |

### 9.4 Teste final do modelo (slide 15)

O slide roda a avaliação na hora, a partir dos scores salvos, com o mesmo `avaliacao.evaluate` do protocolo.

- [ ] **Qual teste entra.** Uma célula escolhe a fonte automaticamente e escreve no título do slide qual foi usada:
  1. Se as AOIs completas tiverem sido processadas: split `test` do Tapajós e `test_gen` (Rondônia), 2023–2025.
     É o resultado oficial.
  2. Se não: **teste limitado** nos 16 patches `test` da sub-AOI, 2023–2025. O slide avisa que a sub-AOI foi
     escolhida olhando os rótulos, que há poucos positivos e que os ICs são largos.
- [ ] **Tabela:** AE, melhor combinação da fusão, melhor baseline espectral e kNN estático; AUROC, AUPRC e
      TPR@1% FPR com IC 95%, para "todos" e por tipo; mostrar `n` e `n_pos`.
- [ ] **Curva ROC** dos mesmos métodos, com a diagonal do chute.
- [ ] **Acurácia no limiar operacional:** limiar escolhido na validação para FPR = 1% (`threshold_at_fpr`),
      aplicado no teste → matriz de confusão, acurácia, precisão e recall.
- [ ] **Por que a acurácia engana aqui:** com ~7% de patch-trimestres com evento, chamar tudo de "normal" já dá
      ~93% de acurácia. Mostrar esse baseline trivial ao lado, como no `03_baseline` (classe majoritária).
- [ ] Uma frase de leitura honesta: se o AE ou a fusão não superam o baseline simples, o slide diz isso.

### 9.5 Limitações (slide 18)

- [ ] **AOIs completas não processadas** (se for o caso na data da apresentação). Mostrar o planejado × o feito:
      Tapajós + Rondônia ≈ 19,4 mil patches e ~545 mil patch-trimestres (~12 h de CPU só no TerraMind, dezenas de
      horas de download do Earth Engine) × sub-AOI de 16 × 16 patches (256 patches, 35,8 km). Consequências:
      - métricas vêm de uma área pequena, escolhida pelos rótulos, com poucos positivos e ICs largos;
      - a normalização local por cobertura não pôde ser testada (todos os normais de validação são floresta);
      - a generalização para Rondônia (E7 / J6) não foi testada;
      - mapas só na sub-AOI.
- [ ] **Infraestrutura:** sem GPU local, ~2 GB livres no Drive, download do Earth Engine como gargalo
      (tabela "Limitações de infraestrutura" do plano principal).
- [ ] **Rótulos:** DETER não pega cortes pequenos; PRODES é anual; parte das "mudanças" do MapBiomas são trocas
      entre classes parecidas (pastagem ↔ mosaico de usos), não mudanças físicas.
- [ ] **Resultado:** no diagnóstico, o AE não superou os baselines simples; no anual, o kNN estático vence porque
      separa "é fronteira?" de "floresta intacta", e não por detectar o evento.
- [ ] **Escala:** o patch de 2,24 km e a célula de 320 m ainda são grandes para eventos de poucos hectares.
- [ ] **Trilha de pastagem:** o rótulo de vigor do MapBiomas é saída de outro modelo, não verdade de campo.

### 9.6 Tarefas

- [x] Esqueleto do notebook com os 20 slides e os metadados de slide.
- [x] Funções de figura reaproveitáveis em `notebooks/gabriel/figuras.py` (RGB, NDVI/BSI, grade de células, mapa
      por célula, série do score, erros). Polígonos por cima ainda não: os slides contornam o **patch** com evento.
- [x] Exemplos fixos em `figuras.EXEMPLOS` (todos em blocos `val`): desmatamento `tapajos_252_4125` 2024Q3, fogo
      `tapajos_244_4120` 2024Q3, garimpo `tapajos_245_4122` 2023Q1.
- [ ] Célula do teste final (9.4) com a escolha automática da fonte. Por ora fixa no **teste limitado** (abaixo);
      trocar quando as AOIs completas existirem.
- [x] Exportar para HTML (`notebooks/apresentacao.slides.html`). **Falta:** testar sem internet (reveal.js vem de CDN)
      e gerar o PDF de reserva.
- [ ] Ensaio: 15–20 min de fala; notas do apresentador em cada slide.
- [x] Adicionar `nbconvert` ao `notebooks/gabriel/requirements-local.txt`.

> **Status (05/11), versão para a apresentação técnica de 06/11:** feita **antes** da Fase 5, porque a apresentação foi
> antecipada. Os números são da sub-AOI e **não são os finais**; os slides dizem isso no título, no teste final e nas
> limitações.
> - **Teste final = teste limitado** (`PROTOCOLO.md`, seção 7, registrado antes de calcular): os 16 patches `test` da
>   sub-AOI não têm nenhum evento em 2023–2025, então usamos os blocos `val` em 2023–2025 (29 eventos, 537 limpos).
> - Resultado (AUROC, todas): queda de NDVI 0,769; kNN no delta TerraMind 0,743; Deep SVDD original 0,692; fusão
>   0,675; AE-TM 0,668; AE-ORIG 0,656; AE-TM+ORIG 0,632. ICs largos e sobrepostos.
> - No limiar de FPR 1% (escolhido nos limpos de val 2021–22), o AE-TM+ORIG marca 0 de 29 eventos: acurácia 94,5% ×
>   94,9% de "tudo normal". A 5%, 1 de 29.
> - Os falsos positivos mais fortes são **garimpo visível** em patches rotulados como limpos (lacuna de rótulo).
> - O notebook roda em ~8 min só com resultados salvos.

---

## 10. Experimentos

| # | Experimento | Pergunta | Liga com |
|---|---|---|---|
| J1 | Fusão no patch (scores salvos) | Vale a pena continuar? | — |
| J2 | Fusão na célula, par principal + ablações | A combinação supera o AE e o melhor espectral? | E1 |
| J3 | Features atuais (14) × originais (20), mesmo detector | A engenharia de features original acrescenta algo? | E3 |
| J4 | AE-TM × AE-ORIG × AE-TM+ORIG | Juntar na entrada é melhor que juntar no score? | E3 |
| J5 | J2 e J4 por tipo de evento | Quem ganha em desmatamento, fogo, degradação, mineração | E8 |
| J6 | Melhor combinação treinada no Tapajós, testada em Rondônia | A combinação generaliza? | E7 |

**Comparação justa.**
- [x] Adicionar a `avaliacao.py` uma comparação **pareada** (`avaliacao.compare`, `utils.block_bootstrap_diff`): diferença de AUROC entre dois métodos, com IC 95% por
      bootstrap dos mesmos blocos em cada reamostragem. Os ICs separados se sobrepõem muito com poucos positivos;
      o pareado é mais sensível.
- [ ] Reportar sempre: AUROC, AUPRC, TPR@1% FPR, por tipo e com IC.
- [ ] A combinação principal é a da Fase 0. Se uma ablação vencer no teste, isso é reportado como achado, sem
      trocar a principal.

---

## 11. Opcional (se sobrar tempo)

- **Covariáveis do notebook 04** (altitude, declividade, chuva CHIRPS): não como entrada do detector, mas como
  estratos extras da normalização local (ex.: várzea × terra firme), via `reduceRegions` no Earth Engine por patch.
- **Caminho inverso, na tarefa de pastagem.** Para a entrega da disciplina (Grupo 1), dá para comparar nos 1.800
  pontos de Goiânia: regressão logística com as 15 features × com o embedding TerraMind × com os dois juntos.
  `embeddings.fetch_points` já baixa um patch 224 × 224 (12 bandas + S1) centrado em cada ponto e extrai o embedding.
  É supervisionado e separado do projeto de anomalias; não usa o AE.

---

## 12. Cronograma

Encaixa nas semanas 4 a 6 do plano principal (entrega no fim de novembro).

| Período | Foco | Gate ao final |
|---|---|---|
| 05–07/11 | Fase 0; Fase 1 (`features_originais.py`, sub-AOI) e integração no `write_step` | Features na sub-AOI; fluxo das AOIs completas pronto para rodar |
| 08–11/11 | Rodar `stream_embeddings.py` à noite; Fase 2 e J1/J2 na sub-AOI | Decisão: seguir com o caminho B? |
| 12–18/11 | Fase 4 na sub-AOI (J3, J4); retreino nas AOIs completas | Modelos finais escolhidos só com normais de validação |
| 19–25/11 | Fase 5: teste no Tapajós e em Rondônia (J5, J6); mapas com o score combinado. Fase 6: esqueleto dos slides e figuras de imagem (slides 1–14), que não dependem do teste final | Tabela final; slides 1–14 prontos |
| 26–28/11 | Fase 6: teste final (9.4), mapas, erros e limitações nos slides; exportar HTML e PDF | Apresentação completa |
| 29–30/11 | Ensaio, texto, README e entrega | **Entrega** |

**Se o tempo apertar, cortar nesta ordem:** opcionais (seção 11) → J4 / caminho B → J2 (ficar com J1) → Fourier →
slides 16–17 (mapa e erros).
**Não cortar** a integração da Fase 1 no `write_step`: sem ela, as features originais nas AOIs completas exigem
baixar tudo de novo. **Não cortar** o teste final (9.4) nem as limitações (9.5) da apresentação: se as AOIs
completas não ficarem prontas, a apresentação usa o teste limitado e diz isso.

---

## 13. Riscos

| Risco | Mitigação |
|---|---|
| Fusão não melhora | Ainda responde o E3 (FM × features clássicas); reportar como resultado |
| Escolher a combinação olhando o teste | Regras fixadas na Fase 0; ablações só como ablação |
| Features originais abafadas no AE combinado | Perda com peso por grupo (Fase 4) |
| Fourier ruidoso em células com nuvem | Célula com < 50% de pixels limpos vira NaN; detectores usam `nan_to_num` só depois da máscara |
| Atrasar o fluxo das AOIs completas | Fase 1 é pequena; se travar, rodar o fluxo sem Fourier e manter só as médias e desvios |
| Diagnóstico da sub-AOI enganoso | Números da sub-AOI nunca são reportados (avaliação no split `train`) |
| AOIs completas não terminam a tempo | A apresentação usa o teste limitado da sub-AOI (9.4) e mostra o planejado × o feito (9.5) |
| Slides travam sem internet ou o notebook demora para rodar | reveal.js local + PDF de reserva; o notebook só lê resultados salvos |
| Números dos slides diferentes dos de `results/` | O teste final é calculado na hora com o mesmo `avaliacao.evaluate`, a partir dos scores salvos |

---

## 14. Arquivos

| Arquivo | Novo / alterado | Conteúdo |
|---|---|---|
| `notebooks/gabriel/features_originais.py` | novo | Features originais por célula (seção 4) |
| `notebooks/gabriel/fusao.py` | novo | Regras de fusão (z robusto, percentis), no patch e na célula |
| `notebooks/gabriel/utils.py` | alterado | `block_bootstrap_diff` (bootstrap pareado) |
| `notebooks/gabriel/embeddings.py` | alterado | `write_step` grava `orig_cells`; `open_cells` cria o array |
| `notebooks/gabriel/baselines.py` | alterado | Grupo "Original" de detectores; leitura de `orig_cells` |
| `notebooks/gabriel/modelo.py` | alterado | `entrada` / `d_in` parametrizáveis; validade por célula; perda por grupo; `Trainer.load` |
| `notebooks/gabriel/avaliacao.py` | alterado | Comparação pareada por bootstrap de blocos |
| `notebooks/gabriel/08_fusao.ipynb` | novo | J1, J2 |
| `notebooks/gabriel/09_ae_combinado.ipynb` | novo | J3, J4 |
| `notebooks/gabriel/PROTOCOLO.md` | alterado | Regras da Fase 0 |
| `notebooks/apresentacao.ipynb` | novo | Slides da apresentação (Fase 6) |
| `notebooks/apresentacao.slides.html` | novo (gerado) | Slides exportados com `nbconvert` |
| `notebooks/gabriel/figuras.py` | novo | Funções de figura usadas nos slides e nos mapas |
| `results/apresentacao/` | novo (gerado) | PNGs das figuras e a tabela do teste final |
| `notebooks/gabriel/requirements-local.txt` | alterado | `nbconvert` (e `rise`, opcional) |
| `notebooks/dataset_utils.py` | inalterado | Referência das fórmulas (NDVI, GNDVI, agregados) |
