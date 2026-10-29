"""Etapa 3: composições Sentinel-2/Sentinel-1 por trimestre, baixadas por bloco espacial e guardadas em Zarr.

Unidade de download = 1 bloco espacial (4 x 4 patches = 896 x 896 px a 10 m) x 1 período x 1 sensor,
via ee.data.computePixels (direto para numpy, sem exportar para o Drive). A grade em pixels coincide
com a grade de patches da Etapa 2 (UTM, múltiplos de 2240 m), então não há reamostragem de posição.

No cubo ficam os valores **brutos** (S2 em DN L2A, S1 em dB); a padronização do TerraMind é feita na
leitura (`normalize`), para poder trocar de modelo sem baixar de novo.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config
import rotulos

B = config.BLOCK_PATCHES
P = config.PATCH_SIZE
BLOCK_PX = B * P                      # 896
S1_NODATA = -9999.0
EE_DEADLINE_MS = 180_000              # sem prazo, uma requisição travada parou um lote por 29 min (29/10)
_deadline_lock = __import__('threading').Lock()
_deadline_set = False


def _ensure_deadline():
    """Prazo por requisição do Earth Engine, definido uma vez (setDeadline reinstala o cliente da API)."""
    global _deadline_set
    with _deadline_lock:
        if not _deadline_set:
            import ee
            ee.data.setDeadline(EE_DEADLINE_MS)   # requisição travada vira erro e cai no retry de fetch_block
            _deadline_set = True


# ---------------- composições (Earth Engine) ----------------

def _region(x0, y0, epsg, size_m):
    import ee
    return ee.Geometry.Rectangle([x0, y0, x0 + size_m, y0 + size_m], f'EPSG:{epsg}', False)


def s2_composite(start, end, region):
    """Mediana das cenas S2 L2A com Cloud Score+ >= CLOUD_SCORE_MIN, 12 bandas do TerraMind + n_obs."""
    import ee
    names = config.S2_BANDS + ['n_obs']
    col = (ee.ImageCollection(config.S2_ASSET).filterBounds(region).filterDate(start, end)
           .linkCollection(ee.ImageCollection(config.CLOUD_SCORE_ASSET), ['cs_cdf'])
           .map(lambda im: im.updateMask(im.select('cs_cdf').gte(config.CLOUD_SCORE_MIN))
                .select(config.S2_BANDS)))
    comp = col.median().unmask(0).addBands(col.select('B4').count().rename('n_obs').unmask(0))
    empty = ee.Image.constant([0] * len(names)).rename(names)
    return ee.Image(ee.Algorithms.If(col.size().gt(0), comp, empty)).select(names).toUint16()


def s1_composite(start, end, region):
    """Mediana (em dB) das cenas S1 GRD IW com VV e VH, órbitas ascendente e descendente + n_obs."""
    import ee
    names = config.S1_BANDS + ['n_obs']
    col = (ee.ImageCollection(config.S1_ASSET).filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .select(config.S1_BANDS))
    comp = col.median().unmask(S1_NODATA).addBands(col.select('VV').count().rename('n_obs').unmask(0))
    empty = ee.Image.constant([S1_NODATA, S1_NODATA, 0]).rename(names)
    return ee.Image(ee.Algorithms.If(col.size().gt(0), comp, empty)).select(names).toFloat()


COMPOSITES = {'s2': s2_composite, 's1': s1_composite}


def fetch_block(sensor, x0, y0, epsg, start, end, retries=5, size_px=BLOCK_PX):
    """Baixa uma janela size_px x size_px (padrão: um bloco, 896 px) com canto inferior esquerdo (x0, y0) em UTM.
    Retorna (dados (C, H, W), n_obs (H, W)); S2 uint16, S1 float32 com NaN onde não há observação."""
    import ee
    _ensure_deadline()
    size_m = size_px * config.RESOLUTION_M
    img = COMPOSITES[sensor](start, end, _region(x0, y0, epsg, size_m))
    req = {'expression': img, 'fileFormat': 'NUMPY_NDARRAY',
           'grid': {'dimensions': {'width': size_px, 'height': size_px},
                    'affineTransform': {'scaleX': config.RESOLUTION_M, 'shearX': 0, 'translateX': x0,
                                        'shearY': 0, 'scaleY': -config.RESOLUTION_M, 'translateY': y0 + size_m},
                    'crsCode': f'EPSG:{epsg}'}}
    for attempt in range(retries):
        try:
            arr = ee.data.computePixels(req)
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(10 * 2 ** attempt + np.random.rand() * 5)   # 429 / concorrência: recua e tenta de novo
    bands = config.S2_BANDS if sensor == 's2' else config.S1_BANDS
    data = np.stack([arr[b] for b in bands])
    nobs = np.minimum(arr['n_obs'], 255).astype(np.uint8)
    if sensor == 's1':
        data = np.where(data <= S1_NODATA + 1, np.nan, data).astype(np.float32)
    return data, nobs


def block_to_patches(a):
    """(..., 896, 896) -> (16, ..., 224, 224), patches na ordem de `block_patch_order` (linha de cima primeiro)."""
    lead = a.shape[:-2]
    t = a.reshape(*lead, B, P, B, P)                       # (..., r, y, c, x); r = 0 é o topo
    t = np.moveaxis(t, (-4, -2), (0, 1))                   # (r, c, ..., y, x)
    return t.reshape(B * B, *lead, P, P)


def block_patch_order(block_patches):
    """Ordena os 16 patches de um bloco como `block_to_patches`: de cima para baixo, da esquerda para a direita."""
    return block_patches.sort_values(['row', 'col'], ascending=[False, True])


# ---------------- normalização TerraMind ----------------

def normalize(x, modality):
    """Padroniza (..., C, H, W) com as estatísticas de pré-treino do TerraMind (float32)."""
    s = config.TERRAMIND_STATS[modality]
    shape = (-1, 1, 1)
    return ((x.astype(np.float32) - np.array(s['mean'], np.float32).reshape(shape))
            / np.array(s['std'], np.float32).reshape(shape))


# ---------------- sub-AOI ----------------

def select_subaoi(grid, eventos, normal, size_blocks=config.SUBAOI_BLOCKS,
                  tipos=('desmatamento', 'mineracao', 'fogo', 'degradacao'), min_test_patches=5):
    """Escolhe a janela de size x size blocos com maior diversidade de eventos.

    Critério: média geométrica de (1 + patches com cada tipo, >= EVENT_MIN_FRAC) no período todo, entre as
    janelas que (a) estão inteiras na AOI, (b) têm 30–70% de normais de treino, (c) têm blocos de val e de
    test e (d) têm pelo menos `min_test_patches` patches com cada tipo em 2023–2025 (período de teste)."""
    ev = eventos[eventos['frac'] >= config.EVENT_MIN_FRAC]
    ev = ev.assign(year=pd.to_datetime(ev['date_end']).dt.year)
    anyp = ev.groupby('tipo')['patch_id'].apply(set).to_dict()
    testp = ev[ev['year'] > config.TRAIN_END_YEAR].groupby('tipo')['patch_id'].apply(set).to_dict()
    g = grid.merge(normal[['patch_id', 'normal_treino']], on='patch_id')
    g = g.assign(bc=g['col'] // B, br=g['row'] // B)
    full = g.groupby(['bc', 'br']).size()
    full = set(full[full == B * B].index)
    rows = []
    for bc, br in sorted(full):
        win = {(bc + i, br + j) for i in range(size_blocks) for j in range(size_blocks)}
        if not win <= full:
            continue
        w = g[[(c, r) in win for c, r in zip(g['bc'], g['br'])]]
        ids = set(w['patch_id'])
        r = {'bc': bc, 'br': br, 'normal': w['normal_treino'].mean(),
             'val': (w['split'] == 'val').any(), 'test': (w['split'] == 'test').any()}
        r.update({t: len(ids & anyp.get(t, set())) for t in tipos})
        r.update({f'{t}_teste': len(ids & testp.get(t, set())) for t in tipos})
        rows.append(r)
    c = pd.DataFrame(rows)
    ok = c['normal'].between(0.3, 0.7) & c['val'] & c['test']
    for t in tipos:
        ok &= c[f'{t}_teste'] >= min_test_patches
    c['score'] = np.exp(np.log1p(c[list(tipos)]).mean(axis=1)).where(ok, 0.0)
    return c.sort_values('score', ascending=False).reset_index(drop=True)


def subaoi_patches(grid, origin=config.SUBAOI_ORIGIN, size_blocks=config.SUBAOI_BLOCKS):
    """Patches da sub-AOI ordenados por bloco (16 consecutivos por bloco), como no cubo."""
    c0, r0 = origin
    n = size_blocks * B
    sub = grid[grid['col'].between(c0, c0 + n - 1) & grid['row'].between(r0, r0 + n - 1)]
    assert len(sub) == n * n, f'sub-AOI incompleta: {len(sub)} patches'
    return (pd.concat([block_patch_order(b) for _, b in sub.groupby('block', sort=True)])
            .reset_index(drop=True))


# ---------------- cubo Zarr ----------------

def open_cube(path, patches, epsg, steps=config.STEPS, mode='a'):
    """Cria (ou abre) o cubo. Patches devem vir de `subaoi_patches` (blocos contíguos)."""
    import zarr
    root = zarr.open_group(str(path), mode=mode)
    if 's2' in root:
        assert list(root.attrs['patch_id']) == list(patches['patch_id']), 'cubo existente com outra lista de patches'
        return root
    n, t = len(patches), len(steps)
    root.attrs.update({'patch_id': list(patches['patch_id']), 'block': list(patches['block']),
                       'split': list(patches['split']), 'x0': [int(v) for v in patches['x0']],
                       'y0': [int(v) for v in patches['y0']], 'epsg': epsg, 'steps': list(steps),
                       's2_bands': config.S2_BANDS, 's1_bands': config.S1_BANDS,
                       'resolution_m': config.RESOLUTION_M, 'cloud_score_min': config.CLOUD_SCORE_MIN,
                       'note': 'valores brutos: S2 DN L2A (uint16, 0 = sem dado), S1 dB (float16, NaN = sem dado)'})
    nb = B * B   # um chunk = 1 bloco x 1 passo
    root.create_array('s2', shape=(n, t, len(config.S2_BANDS), P, P), chunks=(nb, 1, len(config.S2_BANDS), P, P),
                      dtype='uint16', fill_value=0)
    root.create_array('s2_nobs', shape=(n, t, P, P), chunks=(nb, 1, P, P), dtype='uint8', fill_value=0)
    root.create_array('s1', shape=(n, t, len(config.S1_BANDS), P, P), chunks=(nb, 1, len(config.S1_BANDS), P, P),
                      dtype='float16', fill_value=np.nan)
    root.create_array('s1_nobs', shape=(n, t, P, P), chunks=(nb, 1, P, P), dtype='uint8', fill_value=0)
    n_blocks = n // nb
    root.create_array('done', shape=(2, n_blocks, t), chunks=(2, n_blocks, t), dtype='bool', fill_value=False)
    return root


def fill_cube(root, patches, epsg, sensors=('s2', 's1'), workers=8, budget_gb=config.STORAGE_BUDGET_GB,
              cube_path=None, check_after=8):
    """Baixa todos os (sensor, bloco, passo) que faltam, em paralelo. Retomável: pula o que já está em `done`.
    Depois de `check_after` tarefas, projeta o tamanho final pelo tamanho real em disco e para se passar do orçamento."""
    steps = list(root.attrs['steps'])
    nb = B * B
    blocks = [patches.iloc[i:i + nb] for i in range(0, len(patches), nb)]
    done = root['done'][:]
    todo = [(s, b, t) for s in sensors for b in range(len(blocks)) for t in range(len(steps))
            if not done[('s2', 's1').index(s), b, t]]
    total = len(sensors) * len(blocks) * len(steps)
    print(f'{len(todo)} de {total} tarefas pendentes', flush=True)

    def job(s, b, t):
        blk = blocks[b]
        x0, y0 = int(blk['x0'].min()), int(blk['y0'].min())
        start, end = rotulos.step_bounds(steps[t])
        data, nobs = fetch_block(s, x0, y0, epsg, start.strftime('%Y-%m-%d'),
                                 (end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        sl = slice(b * nb, (b + 1) * nb)
        root[s][sl, t] = block_to_patches(data).astype(root[s].dtype)
        root[f'{s}_nobs'][sl, t] = block_to_patches(nobs)
        return s, b, t

    t0, finished = time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(job, *k) for k in todo]
        for f in as_completed(futs):
            s, b, t = f.result()
            root['done'][('s2', 's1').index(s), b, t] = True
            finished += 1
            if finished % 20 == 0 or finished == len(todo):
                el = time.time() - t0
                print(f'  {finished}/{len(todo)}  {el / 60:.1f} min  (~{el / finished * (len(todo) - finished) / 60:.0f} min restantes)',
                      flush=True)
            if cube_path is not None and finished == check_after:
                n_done = int(root['done'][:].sum())
                proj = disk_gb(cube_path) / max(n_done, 1) * total
                print(f'  projeção do cubo: {proj:.1f} GB (orçamento {budget_gb} GB)', flush=True)
                if proj > budget_gb:
                    for g in futs:
                        g.cancel()
                    raise RuntimeError(f'projeção {proj:.1f} GB > orçamento {budget_gb} GB: reduza a sub-AOI')


def disk_gb(path):
    from pathlib import Path
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file()) / 1e9


def clear_fraction(root, sensor='s2'):
    """(patch, passo): fração de pixels com pelo menos uma observação limpa."""
    nobs = root[f'{sensor}_nobs']
    out = np.empty(nobs.shape[:2], np.float32)
    for t in range(nobs.shape[1]):
        out[:, t] = (nobs[:, t] > 0).mean(axis=(-2, -1))
    return out
