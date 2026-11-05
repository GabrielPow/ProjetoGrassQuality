"""Features da implementação original (notebooks/dataset_utils.py e 07_fourier_mudanca.ipynb) na célula de 320 m.

Plano: PLANO_JUNTAR_DEEP_LEARNING.md (Fase 1); regras fixadas em PROTOCOLO.md, seção 6.

Para cada célula (32 x 32 px, a mesma grade 7 x 7 das células do TerraMind), 13 features que ainda não existiam
nas células do Gabriel:
- desvio de B2, B3, B4, B8, B11, B12 (textura; reflectância 0–1);
- desvio do NDVI; médias de GNDVI e BSI;
- log10 da energia do espectro de Fourier do NDVI em 4 anéis de frequência (janela de Hann, média removida).

Junto com as médias de B2, B3, B4, B8, B11, B12 e NDVI de `spec_cells` (embeddings.cell_spectral), formam o
conjunto "Original" de 20 features (`original_features`). Célula com < 50% de pixels limpos fica NaN.
"""
import numpy as np

import config

ORIG_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']          # as 6 bandas de dataset_utils.BAND_NAMES
RING_EDGES = [0.0, 0.04, 0.08, 0.125, np.inf]                 # ciclos/pixel; o anel externo usa o corte do 07
ORIG_CELL_NAMES = ([f'{b}_desvio' for b in ORIG_BANDS] + ['NDVI_desvio', 'GNDVI_media', 'BSI_media']
                   + [f'FFT_NDVI_anel{k + 1}' for k in range(len(RING_EDGES) - 1)])
# colunas de spec_cells (embeddings.CELL_SPEC_NAMES) que entram no conjunto Original
SPEC_COLS = [config.S2_BANDS.index(b) for b in ORIG_BANDS] + [12]           # 12 = NDVI_media
ORIGINAL_NAMES = [f'{b}_media' for b in ORIG_BANDS] + ['NDVI_media'] + ORIG_CELL_NAMES
HIGH_RINGS = [len(ORIG_CELL_NAMES) - 2, len(ORIG_CELL_NAMES) - 1]           # anéis 3 e 4 (> 0,08 ciclo/pixel)
MIN_CELL_CLEAR = 0.5
EPS_LOG = 1e-8

_G = config.CELL_GRID                       # 7
_C = config.PATCH_SIZE // _G                # 32
_HANN = np.outer(np.hanning(_C), np.hanning(_C)).astype(np.float32)
_FY, _FX = np.meshgrid(np.fft.fftfreq(_C), np.fft.fftfreq(_C), indexing='ij')
_R = np.hypot(_FY, _FX)
_RINGS = [(_R > lo) & (_R <= hi) for lo, hi in zip(RING_EDGES[:-1], RING_EDGES[1:])]   # exclui a frequência zero


def _to_cells(a):
    """(N, ..., 224, 224) -> (N, ..., 49, 32, 32), células linha a linha a partir do topo (igual a pool_cells)."""
    lead = a.shape[:-2]
    a = a.reshape(*lead, _G, _C, _G, _C)
    a = np.moveaxis(a, -3, -2)                                     # (..., 7, 7, 32, 32)
    return a.reshape(*lead, _G * _G, _C, _C)


def _indices(x):
    """x: (N, 12, H, W) reflectância -> NDVI, GNDVI, BSI (N, H, W). Fórmulas de dataset_utils e do notebook 07."""
    b = {n: x[:, config.S2_BANDS.index(n)] for n in ['B2', 'B3', 'B4', 'B8', 'B11']}
    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi = (b['B8'] - b['B4']) / (b['B8'] + b['B4'] + 1e-6)                      # Rouse et al., 1974
        gndvi = (b['B8'] - b['B3']) / (b['B8'] + b['B3'] + 1e-6)                     # Gitelson et al., 1996
        bsi = (((b['B11'] + b['B4']) - (b['B8'] + b['B2']))
               / ((b['B11'] + b['B4']) + (b['B8'] + b['B2']) + 1e-6))
    return ndvi, gndvi, bsi


def ring_energy(ndvi_cells, mask_cells):
    """(..., 32, 32) NDVI + máscara -> (..., 4): log10 da energia média de |FFT|² em cada anel.
    Pixels sem observação recebem a média da célula; a média é removida antes da janela de Hann."""
    m = mask_cells.astype(np.float32)
    cnt = m.sum(axis=(-2, -1), keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = (np.nan_to_num(ndvi_cells) * m).sum(axis=(-2, -1), keepdims=True) / cnt
    a = np.where(m > 0, ndvi_cells, mu) - mu
    a = np.nan_to_num(a) * _HANN
    p = np.abs(np.fft.fft2(a)) ** 2 / (_HANN ** 2).sum()           # energia por frequência, em unidades de variância
    e = np.stack([p[..., r].mean(-1) for r in _RINGS], axis=-1)
    return np.log10(e + EPS_LOG)


def cell_features_originais(s2, nobs):
    """(N, 12, 224, 224) uint16 + (N, 224, 224) -> (N, 49, 13) float32 (ordem ORIG_CELL_NAMES)."""
    x = s2.astype(np.float32) / 10000
    mask = _to_cells(nobs > 0)                                      # (N, 49, 32, 32)
    m = mask.astype(np.float32)
    cnt = m.sum(axis=(-2, -1))                                      # (N, 49)
    ok = cnt / (_C * _C) >= MIN_CELL_CLEAR

    def masked_stats(v):                                            # v: (N, 49, 32, 32)
        with np.errstate(invalid='ignore', divide='ignore'):
            vz = np.nan_to_num(v)
            mu = (vz * m).sum(axis=(-2, -1)) / cnt
            sd = np.sqrt((((vz - mu[..., None, None]) ** 2) * m).sum(axis=(-2, -1)) / cnt)
        return mu, sd

    feats = []
    for b in ORIG_BANDS:
        feats.append(masked_stats(_to_cells(x[:, config.S2_BANDS.index(b)]))[1])
    ndvi, gndvi, bsi = (_to_cells(v) for v in _indices(x))
    feats.append(masked_stats(ndvi)[1])
    feats.append(masked_stats(gndvi)[0])
    feats.append(masked_stats(bsi)[0])
    out = np.concatenate([np.stack(feats, axis=-1), ring_energy(ndvi, mask)], axis=-1)
    out[~ok] = np.nan
    return out.astype(np.float32)


def original_features(spec_cells, orig_cells):
    """Junta as 7 colunas de spec_cells com as 13 de orig_cells -> (..., 20) na ordem ORIGINAL_NAMES."""
    return np.concatenate([spec_cells[..., SPEC_COLS], orig_cells], axis=-1)


def texture_change(orig_t, orig_hist):
    """Mudança de textura (Fourier): soma de (log E(t) − log E_hist)² nos anéis 3 e 4. NaN se faltar dado."""
    d = orig_t[..., HIGH_RINGS] - orig_hist[..., HIGH_RINGS]
    return (d ** 2).sum(-1)


# ---------------- sub-AOI: a partir do cubo ----------------

def orig_from_cube(cube_path, out_path, log_every=4):
    """Calcula `orig_cells` de todos os passos do cubo e grava no zarr de células (sem rodar o TerraMind).
    A ordem dos slots do cubo é a mesma das células (embeddings.extract_cube). Retomável por passo."""
    import time
    import zarr
    import embeddings
    cube = zarr.open_group(str(cube_path), mode='r')
    n, T = cube['s2'].shape[:2]
    cells = embeddings.open_cells(out_path, n, T)
    arr = cells['orig_cells']
    done = cells.attrs.get('orig_done', [False] * T)
    t0 = time.time()
    for t in range(T):
        if done[t]:
            continue
        for b0 in range(0, n, 64):                                  # 64 patches por vez: ~150 MB de pixels
            sl = slice(b0, min(b0 + 64, n))
            arr[sl, t] = cell_features_originais(cube['s2'][sl, t], cube['s2_nobs'][sl, t]).astype(np.float16)
        done[t] = True
        cells.attrs['orig_done'] = done
        if (t + 1) % log_every == 0:
            print(f'  passo {t + 1}/{T}  {(time.time() - t0) / 60:.1f} min', flush=True)
    return arr
