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
AOI_GENERALIZATION = None               # TODO: sul do AM (Lábrea/Apuí) ou norte de RO (decisão em aberto)

# ---- tempo ----
YEARS = list(range(2019, 2026))
PERIOD = 'quarterly'                    # 4 passos/ano
TRAIN_END_YEAR = 2022                   # treino: sequências terminando até 2022; teste: 2023–2025

# ---- sensores e patches ----
S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']   # conferir contra o que o TerraMind S2L2A espera (Etapa 3)
S1_BANDS = ['VV', 'VH']
PATCH_SIZE = 224
RESOLUTION_M = 10
SPATIAL_BLOCK_KM = 10

# ---- classes MapBiomas: reaproveitadas do first pass (common.py só importa ee, não inicializa) ----
sys.path.insert(0, str(FIRSTPASS_DIR))
from common import FOREST, PASTURE, MINING, ANTHROPIC, LULC_ASSET  # noqa: E402


def as_dict():
    """Config serializável, para salvar junto aos resultados de cada execução."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in globals().items()
            if k.isupper()}


def make_dirs():
    for d in (DATA_DIR, EMBEDDINGS_DIR, CHECKPOINTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
