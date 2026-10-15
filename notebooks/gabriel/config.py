"""Config compartilhada dos notebooks de detecção de anomalias (PLANO_IMPLEMENTACAO_ANOMALIA.md).

Roda localmente (Windows, sem GPU) e no Google Colab (GPU, dados no Drive).
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
S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']   # conferir contra o que o TerraMind S2L2A espera (Etapa 3)
S1_BANDS = ['VV', 'VH']
PATCH_SIZE = 224
RESOLUTION_M = 10
PATCH_M = PATCH_SIZE * RESOLUTION_M     # 2240 m; grade alinhada a múltiplos de 2240 m em UTM
BLOCK_PATCHES = 4                       # bloco espacial = 4 x 4 patches = 8,96 km
SPLIT_TARGET = (0.70, 0.15, 0.15)       # train / val / test, em número de patches (AOI principal)

# ---- rótulos (Etapa 2) ----
EVENT_MIN_FRAC = 0.01                   # >= 1% do patch (~5 ha) conta como evento na avaliação
# 0 < fração < EVENT_MIN_FRAC: patch-passo "ambíguo", fora da avaliação e fora dos normais
NORMAL_MAX_FRAC = 0.001                 # normal de treino: em cada fonte, soma 2019–2025 < 0,1% (~0,5 ha);
                                        # tolera o ruído de pixel isolado (30 m) de MapBiomas/Hansen
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
    for d in (DATA_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LABELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
