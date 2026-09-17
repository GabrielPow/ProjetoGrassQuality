2 Entregáveis comuns a todos os grupos
E1. Dados tratados. Arquivo(s) em data/processed/ (formato .csv, .parquet ou .npz)
gerado(s) por código versionado. Se o volume for grande, entregar uma amostra
representativa e o script que reconstrói o conjunto completo.
E2. Script ou notebook de ingestão e tratamento (01_dados.ipynb ou src/dados.py),
executável do início ao fim, documentando: fonte, período/recorte, licença de
uso, unidades, tratamento de ausentes, remoção de duplicatas e critério de
rotulagem.
E3. Notebook de caracterização (02_caracterizacao.ipynb) com estatísticas descri
tivas.
E4. Notebook de baseline (03_baseline.ipynb) com a partição dos dados, o treinamento,
as métricas e a comparação obrigatória contra um baseline trivial (ver seção de
cada grupo).
E5. README.md com: como reproduzir (ordem de execução, versões de bibliotecas, tempo
aproximado), divisão de tarefas entre os dois integrantes e declaração de uso de
ferramentas de IA generativa (o que foi usado e para quê). Além disso uma breve
(BREVE, 2 páginas no máximo) discussão sobre o Problema, Dados, Caracterização,
Protocolo experimental, Baseline e resultados, Limitações, Trabalhos futuros.



3 Estrutura da avaliação
Os três grupos são avaliados sobre os mesmos cinco blocos, com os mesmos pesos. O
conteúdo específico de cada bloco está detalhado na seção do respectivo grupo.
Bloco
Foco
Pontos
B1. Dados e tratamento - 25 - Obtenção, limpeza, rastreabilidade
B2. Caracterização - 20 - Estatísticas descritivas e figuras
B3. Baseline e protocolo - 30 - Partição, modelo, treino correto
B4. Avaliação e interpretação - 15 - Métricas, comparação, diagnóstico
B5. Trabalhos futuros e reprodutibilidade - 10 - Plano justificado e código executável

4 Grupo 1– Classificação de qualidade de gramado por imagens de
satélite e aéreas
Escopo
Definir operacionalmente o que será chamado de “qualidade de gramado”, montar um
conjunto de imagens (ou recortes) rotulado a partir de fontes abertas e treinar
um classificador simples. Como não existe base pública rotulada diretamente por
qualidade de grama, o grupo deve adotar um rótulo proxy nesta etapa e declará-lo
explicitamente. Duas opções aceitáveis:
(a) Classificação de cobertura usando classes já rotuladas de uma base pública
(por exemplo, distinguir Pasture e HerbaceousVegetation das demais classes do
EuroSAT), tratando o problema como reconhecimento de cobertura vegetal herbácea;
(b) Rótulo por índice espectral, discretizando o NDVI médio do recorte em faixas (por
exemplo, baixa/média/alta densidade de vegetação), com os limiares definidos
pelos quantis do próprio conjunto e justificados no relatório.
Fontes sugeridas (todas públicas)
• EuroSAT– recortes Sentinel-2, 10 classes de uso e cobertura do solo (Helber et
al., 2019).
• UC Merced Land Use– imagens aéreas, 21 classes (Yang e Newsam, 2010).
• Sentinel-2 L2A via Copernicus Data Space ou Microsoft Planetary Computer (STAC).
• Landsat 8/9 Collection 2 via USGS EarthExplorer.
• DeepGlobe Land Cover (Demir et al., 2018), se for optado por segmentação como
trabalho futuro.
Índices espectrais admitidos
Se forem usadas bandas multiespectrais, os índices devem ser calculados pelas
definições publicadas, citando a fonte:
NDVI = ρNIR −ρRED
ρNIR +ρRED
GNDVI = ρNIR −ρGREEN
ρNIR +ρGREEN
(Rouse et al., 1974),
(Gitelson et al., 1996),
onde ρb ∈ [0,1] é a reflectância de superfície na banda b. Para o Sentinel-2,
NIR = B8, RED=B4, GREEN=B3.
3
IBM8924– Projeto de Deep Learning
AC de Projeto– 2026.2
Requisitos mínimos por bloco
Bloco
Requisito mínimo
Pts
B1. Dados e
tratamento
Mínimo de 2 fontes consultadas e ao menos 1 efetivamente
processada. Conjunto final com ≥ 1500 recortes, dimensão
fixa (por exemplo 64 × 64 ou 128 × 128), tensor X ∈
RN× H× W× C com N, H, W, C declarados. Rótulo proxy
definido, justificado e com contagem por classe. Licença
de cada fonte citada.
25
B2. Caracterização
Distribuição de classes; estatísticas por banda (média,
desvio, mínimo, máximo); histograma do índice espectral
usado; grade de exemplos por classe; comentário sobre
desbalanceamento e sobre artefatos (nuvem, sombra, sa
zonalidade).
20
B3. Baseline e
protocolo
Partição estratificada 70/15/15 com semente fixa e sem
recortes da mesma cena em partições diferentes. Dois
modelos obrigatórios: (i) baseline trivial– classe ma
joritária; (ii) baseline principal– regressão logística
sobre atributos agregados (médias e desvios por banda +
NDVI) ou CNN pequena treinada do zero ou transfer lear
ning com ResNet-18/EfficientNet-B0 congelada. Registro
de hiperparâmetros e curva de perda.
30
B4. Avaliação e
interpretação
Acurácia, F1 macro, matriz de confusão no conjunto de
teste e comparação explícita com o baseline trivial.
Análise de ao menos 4 erros (imagens mostradas e comen
tadas).
15
B5. Futuros e
reprodutibilidade
Três direções justificadas (por exemplo: rótulo real
de qualidade via inspeção de campo, séries temporais
NDVI, segmentação, aumento de dados geoespacial) e código
executável.
10
Fora do escopo desta AC
Segmentação semântica, ajuste fino completo de redes profundas, busca de hiperparâ
metros e validação cruzada espacial. Esses itens devem aparecer apenas em Trabalhos
futuros.