"""Funções compartilhadas pelos notebooks 02_caracterizacao.ipynb e 03_baseline.ipynb.

Mantidas fora dos notebooks para evitar duplicar a mesma lógica de cálculo de
índices espectrais / composição RGB / atributos agregados em dois lugares.
"""
import numpy as np

BAND_NAMES = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
BAND_IDX = {b: i for i, b in enumerate(BAND_NAMES)}

# Sentinel-2 SR (COPERNICUS/S2_SR_HARMONIZED) reporta refletância escalada por 10000.
SR_SCALE = 10000.0


def load_dataset(path='../data/processed/s2_patches.npz'):
    """Carrega o dataset salvo por 01_dados.ipynb como um dict de arrays numpy."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def compute_ndvi_gndvi(patch):
    """patch: (H, W, C) na ordem BAND_NAMES. Retorna (ndvi, gndvi), cada (H, W).

    NDVI = (NIR - RED) / (NIR + RED)          (Rouse et al., 1974)
    GNDVI = (NIR - GREEN) / (NIR + GREEN)     (Gitelson et al., 1996)
    Sentinel-2: NIR=B8, RED=B4, GREEN=B3.
    """
    red = patch[..., BAND_IDX['B4']].astype(np.float32) / SR_SCALE
    green = patch[..., BAND_IDX['B3']].astype(np.float32) / SR_SCALE
    nir = patch[..., BAND_IDX['B8']].astype(np.float32) / SR_SCALE
    ndvi = (nir - red) / (nir + red + 1e-6)
    gndvi = (nir - green) / (nir + green + 1e-6)
    return ndvi, gndvi


def rgb_composite(patch, gain=3.5):
    """Composição RGB (B4,B3,B2) normalizada em [0,1] para visualização."""
    rgb = patch[..., [BAND_IDX['B4'], BAND_IDX['B3'], BAND_IDX['B2']]].astype(np.float32) / SR_SCALE
    return np.clip(rgb * gain, 0, 1)


FEATURE_NAMES = (
    [f'{b}_{stat}' for b in BAND_NAMES for stat in ('mean', 'std')]
    + ['NDVI_mean', 'NDVI_std', 'GNDVI_mean']
)


def aggregate_features(patch):
    """Vetor de 15 atributos por patch: média+desvio de cada banda + estatísticas de NDVI/GNDVI.

    Usado como entrada do baseline raso (regressão logística) em 03_baseline.ipynb.
    """
    stats = []
    for i in range(patch.shape[-1]):
        band = patch[..., i].astype(np.float32)
        stats.extend([band.mean(), band.std()])
    ndvi, gndvi = compute_ndvi_gndvi(patch)
    stats.extend([ndvi.mean(), ndvi.std(), gndvi.mean()])
    return np.array(stats, dtype=np.float32)


def aggregate_features_batch(X):
    """X: (N, H, W, C) -> (N, len(FEATURE_NAMES))."""
    return np.stack([aggregate_features(patch) for patch in X])
