"""Config compartilhada dos notebooks de detecção de anomalias (PLANO_IMPLEMENTACAO_ANOMALIA.md).

Desde 29/10 o fluxo principal é **local** (Windows, CPU; TerraMind Small a ~40 ms/patch), porque
restam só 2 GB no Google Drive. O Colab continua possível, mas os pixels vão para o disco temporário da
VM e nunca para o Drive. Ver "Limitações de infraestrutura" no PLANO.
"""
import sys
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ---- caminhos ----
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/content/drive/MyDrive/ProjetoGrassQuality') if IN_COLAB else REPO_ROOT
DATA_DIR = DATA_ROOT / 'data'
EMBEDDINGS_DIR = DATA_ROOT / 'embeddings'
CHECKPOINTS_DIR = DATA_ROOT / 'checkpoints'
RESULTS_DIR = DATA_ROOT / 'results'
# Pixels (cubo Zarr) são um cache descartável, refeito a partir do Earth Engine. Nunca vão para o Drive.
CUBE_DIR = Path('/content/cache') if IN_COLAB else DATA_DIR
FIRSTPASS_DIR = REPO_ROOT / 'experiments' / 'amazon_anomaly_firstpass'

# ---- reprodutibilidade / Earth Engine (mesmos valores do first pass) ----
SEED = 42
EE_PROJECT_ID = 'projetograssquality'

# ---- áreas de estudo: [lon_min, lat_min, lon_max, lat_max] ----
AOI_MAIN = [-57.0, -7.5, -55.0, -5.5]   # Tapajós / BR-163 (PA)
AOI_GENERALIZATION = [-64.0, -10.0, -62.0, -8.0]   # norte de Rondônia (decidido em 15/10); só teste
AOIS = {
    'tapajos':  {'bounds': AOI_MAIN, 'epsg': 32721, 'role': 'principal'},          # UTM 21S
    'norte_ro': {'bounds': AOI_GENERALIZATION, 'epsg': 32720, 'role': 'generalizacao'},  # UTM 20S
}

# ---- tempo ----
YEARS = list(range(2019, 2026))
PERIOD = 'quarterly'                    # 4 passos/ano
STEPS = [f'{y}Q{q}' for y in YEARS for q in range(1, 5)]   # 28 passos, 2019Q1..2025Q4
TRAIN_END_YEAR = 2022                   # treino: sequências terminando até 2022; teste: 2023–2025

# ---- sensores e patches ----
# ordem exata do TerraMind (terratorch PRETRAINED_BANDS['untok_sen2l2a@224']): B1–B12 sem B10
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
S1_BANDS = ['VV', 'VH']                 # dB, como o S1GRD do TerraMind
PATCH_SIZE = 224
RESOLUTION_M = 10
PATCH_M = PATCH_SIZE * RESOLUTION_M     # 2240 m; grade alinhada a múltiplos de 2240 m em UTM
BLOCK_PATCHES = 4                       # bloco espacial = 4 x 4 patches = 8,96 km
SPLIT_TARGET = (0.70, 0.15, 0.15)       # train / val / test, em número de patches (AOI principal)

# ---- rótulos (Etapa 2) ----
EVENT_MIN_FRAC = 0.01                   # >= 1% do patch (~5 ha) conta como evento na avaliação
# 0 < fração < EVENT_MIN_FRAC: patch-passo "ambíguo", fora da avaliação e fora dos normais
NORMAL_MAX_FRAC = 0.001                 # regra ANTIGA (só informativa): em cada fonte, soma 2019–2025 < 0,1%
# Regra em vigor (29/10): normal por JANELA. (patch, ano T) é normal se T-1, T e T+1 não têm evento.
# Motivo: a regra por patch deixava 1 patch de pastagem no treino do Tapajós (ver PROTOCOLO.md).
NORMAL_WINDOW_YEARS = 2                 # a janela de histórico do modelo cabe em T-1..T (até 8 trimestres)
NORMAL_BUFFER_YEARS = 1                 # T+1 também limpo: evita treinar com a véspera de um evento
K_HIST = 4                              # baselines: histórico = média dos 4 trimestres anteriores
EVENT_TYPES = ['desmatamento', 'mineracao', 'fogo', 'degradacao', 'corte_seletivo', 'agua']
DETER_CLASSES = {
    'DESMATAMENTO_CR': 'desmatamento', 'DESMATAMENTO_VEG': 'desmatamento',
    'MINERACAO': 'mineracao', 'CICATRIZ_DE_QUEIMADA': 'fogo', 'DEGRADACAO': 'degradacao',
    'CS_DESORDENADO': 'corte_seletivo', 'CS_GEOMETRICO': 'corte_seletivo',
}
TERRABRASILIS_WFS = 'https://terrabrasilis.dpi.inpe.br/geoserver'
PRODES_LAYER = ('prodes-legal-amz', 'yearly_deforestation')
DETER_LAYER = ('deter-amz', 'deter_amz')
HANSEN_ASSETS = ['UMD/hansen/global_forest_change_2025_v1_13',
                 'UMD/hansen/global_forest_change_2024_v1_12']   # usa o primeiro que existir
MCD64_ASSET = 'MODIS/061/MCD64A1'

# ---- TerraMind (confirmado no Colab em 15/10, terratorch 1.2.13) ----
TERRAMIND_SMALL = 'terratorch_terramind_v1_small'
TERRAMIND_S2_CHANNELS = 12      # S2L2A: B1–B12 sem B10
TERRAMIND_EMBED_DIM = 384
TERRAMIND_TOKEN_GRID = 14       # 224 px / 16 px -> 14 x 14 = 196 tokens
# Unidade de análise (decidida em 29/10 após o gate): CÉLULA = 2 x 2 tokens = 320 m; 7 x 7 = 49 células por patch.
# Os modelos pontuam cada célula; o score do patch é o MÁXIMO das células (rótulos e protocolo ficam no patch).
# Motivo: a média do patch de 2,24 km dilui eventos pequenos (gate: 0,57 no patch vs 0,84 no token).
CELL_POOL = 2
CELL_GRID = TERRAMIND_TOKEN_GRID // CELL_POOL    # 7
N_CELLS = CELL_GRID ** 2                          # 49
CELL_M = PATCH_M // CELL_GRID                     # 320 m
# O backbone ViT NÃO normaliza a entrada: padronizar com as estatísticas de pré-treino
# (terratorch v1.2.13, terramind_register.py, v1_pretraining_mean/std). S2 em DN L2A (refl. x 10000), S1 em dB.
TERRAMIND_STATS = {
    'S2L2A': {'mean': [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011, 3083.234,
                       3132.22, 3162.988, 2424.884, 1857.648],
              'std': [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918,
                      1753.829, 1797.379, 1434.261, 1334.311]},
    'S1GRD': {'mean': [-12.599, -20.293], 'std': [5.195, 5.890]},
}

# ---- dados Sentinel (Etapa 3) ----
S2_ASSET = 'COPERNICUS/S2_SR_HARMONIZED'            # sem o offset de +1000 do baseline 04.00
CLOUD_SCORE_ASSET = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
CLOUD_SCORE_MIN = 0.6                                # mesmo limiar do first pass
S1_ASSET = 'COPERNICUS/S1_GRD'
MIN_CLEAR_FRAC = 0.5    # patch-passo com < 50% de pixels com observação limpa = "dado faltante" (máscara)
# Sub-AOI com cubo de pixels completo (4 x 4 blocos = 16 x 16 patches = 35,8 km), escolhida em
# 03_cubo_sentinel.ipynb por diversidade de eventos. O resto das AOIs vai direto para embeddings (Etapa 4).
SUBAOI_BLOCKS = 4
# (col, row) do patch no canto inferior esquerdo. Escolhido em 29/10 com dados.select_subaoi usando só
# PRODES + DETER (melhor de 30 janelas válidas): lon -56,57 a -56,26, lat -7,13 a -6,82; 256 patches
# (192 train, 48 val, 16 test); 113 patches com desmatamento, 67 mineração, 72 fogo, 35 degradação.
SUBAOI_ORIGIN = (244, 4112)
STORAGE_BUDGET_GB = 20   # o download para se a projeção do cubo passar disso (disco local: 360 GB livres)
CUBE_PATH_NAME = 'cubo_tapajos_sub.zarr'

# ---- classes MapBiomas: reaproveitadas do first pass (common.py só importa ee, não inicializa) ----
sys.path.insert(0, str(FIRSTPASS_DIR))
from common import FOREST, PASTURE, MINING, ANTHROPIC, LULC_ASSET  # noqa: E402

NATURAL_FOREST = [3, 4, 5, 6]           # floresta, savana, mangue, floresta alagável
WATER = [33]
# cobertura dominante do patch (estratificação e normalização local do score, Etapa 6.4)
COVER_GROUPS = {
    'floresta': NATURAL_FOREST,
    'pastagem': PASTURE,
    'agricultura': [9, 18, 19, 20, 36, 39, 40, 41, 46, 47, 48, 62],
    'mosaico': [21],
    'agua': WATER,
    'natural_nao_florestal': [11, 12, 13, 29, 50],
    'outros': [22, 23, 24, 25, 30],
}

# Diretório de rótulos (grade, eventos, rótulos trimestrais) de cada AOI
LABELS_DIR = DATA_ROOT / 'rotulos'


def as_dict():
    """Config serializável, para salvar junto aos resultados de cada execução."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in globals().items()
            if k.isupper()}


def make_dirs():
    for d in (DATA_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LABELS_DIR, CUBE_DIR):
        d.mkdir(parents=True, exist_ok=True)
