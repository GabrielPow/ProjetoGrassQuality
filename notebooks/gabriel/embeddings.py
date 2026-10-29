"""Etapa 4: embeddings do TerraMind Small (congelado) por patch, célula e trimestre.

Unidade de análise (desde 29/10): CÉLULA de 2 x 2 tokens = 320 m, 7 x 7 = 49 por patch (ver config.py).

Layout por "slots": os patches ficam agrupados por bloco espacial, 16 slots por bloco (slot vazio =
patch_id ''), para que cada (bloco, trimestre) baixado seja exatamente um chunk em disco. A sub-AOI
(cubo) já está nessa ordem.

`embeddings/<nome>.npz` (pequeno, em memória):
    patch_id (S,), steps (T,)
    z_s2, z_s2s1   (S, T, 384) float16   média dos 196 tokens da última camada (S2 só / S2 + S1)
    spec           (S, T, 28)  float32   features espectrais do patch (médias e desvios, NDVI, NBR)
    clear_s2, clear_s1 (S, T)  float16   fração de pixels com observação
    done           (S, T)      bool
`embeddings/<nome>_cells.zarr` (grande, em disco):
    c_s2, c_s2s1   (S, T, 49, 384) float16   embedding de cada célula de 320 m
    spec_cells     (S, T, 49, 14)  float16   médias das 12 bandas + NDVI + NBR por célula
A grade completa de tokens (S, T, 196, 384) só é guardada na sub-AOI (`<nome>_tokens.zarr`).
"""
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd

import config
import dados
import rotulos

CONFIGS = {'s2': ['S2L2A'], 's2s1': ['S2L2A', 'S1GRD']}
D = config.TERRAMIND_EMBED_DIM
NB = config.BLOCK_PATCHES ** 2              # 16 slots por bloco
SPEC_NAMES = ([f'{b}_media' for b in config.S2_BANDS] + [f'{b}_desvio' for b in config.S2_BANDS]
              + ['NDVI_media', 'NDVI_desvio', 'NBR_media', 'NBR_desvio'])
CELL_SPEC_NAMES = [f'{b}_media' for b in config.S2_BANDS] + ['NDVI_media', 'NBR_media']


# ---------------- modelo ----------------

def load_model(threads=8):
    import torch
    from terratorch.registry import BACKBONE_REGISTRY
    torch.set_num_threads(threads)
    m = BACKBONE_REGISTRY.build(config.TERRAMIND_SMALL, pretrained=True, modalities=['S2L2A', 'S1GRD']).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def model_inputs(s2, s2_nobs, s1=None, s1_nobs=None):
    """Padroniza com as estatísticas do TerraMind. Pixel sem observação -> 0 (= média do pré-treino)."""
    x2 = dados.normalize(s2, 'S2L2A')
    x2 *= (s2_nobs > 0)[:, None]
    out = {'S2L2A': x2}
    if s1 is not None:
        x1 = np.nan_to_num(dados.normalize(s1, 'S1GRD'), nan=0.0)
        x1 *= (s1_nobs > 0)[:, None]
        out['S1GRD'] = x1
    return out


def pool_cells(tok):
    """(..., 196, D) -> (..., 49, D): média de 2 x 2 tokens. Ordem das células: linha a linha, do topo à esquerda."""
    g, p = config.TERRAMIND_TOKEN_GRID, config.CELL_POOL
    lead, d = tok.shape[:-2], tok.shape[-1]
    t = tok.reshape(*lead, g // p, p, g // p, p, d).astype(np.float32)
    return t.mean(axis=(-4, -2)).reshape(*lead, config.N_CELLS, d)


def embed(model, inputs, batch=32, tokens=False):
    """inputs: {modalidade: (N, C, 224, 224)}. Retorna z (N, D), células (N, 49, D) float16 e, se pedido, tokens."""
    import torch
    n = len(next(iter(inputs.values())))
    zs, cs, toks = [], [], []
    with torch.inference_mode():
        for i in range(0, n, batch):
            out = model({k: torch.from_numpy(np.ascontiguousarray(v[i:i + batch])) for k, v in inputs.items()})[-1]
            out = out.numpy()
            zs.append(out.mean(axis=1))
            cs.append(pool_cells(out).astype(np.float16))
            if tokens:
                toks.append(out.astype(np.float16))
    return np.concatenate(zs), np.concatenate(cs), (np.concatenate(toks) if tokens else None)


def embed_all_configs(model, s2, s2_nobs, s1, s1_nobs, tokens=False):
    inp = model_inputs(s2, s2_nobs, s1, s1_nobs)
    return {name: embed(model, {m: inp[m] for m in mods}, tokens=tokens) for name, mods in CONFIGS.items()}


# ---------------- features espectrais ----------------

def _ndvi_nbr(x):
    b = {n: x[:, i] for i, n in enumerate(config.S2_BANDS)}
    return [(b[a] - b[c]) / (b[a] + b[c] + 1e-6) for a, c in [('B8', 'B4'), ('B8', 'B12')]]


def spectral_features(s2, nobs):
    """(N, 12, H, W) uint16 + (N, H, W) -> (N, 28): média e desvio por banda (reflectância 0–1), NDVI, NBR."""
    x = s2.astype(np.float32) / 10000
    m = (nobs > 0)[:, None]
    cnt = m.sum(axis=(-2, -1))
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = (x * m).sum(axis=(-2, -1)) / cnt
        std = np.sqrt(((x - mean[..., None, None]) ** 2 * m).sum(axis=(-2, -1)) / cnt)
        idx = []
        mm = m[:, 0]
        for v in _ndvi_nbr(x):
            mu = (v * mm).sum(axis=(-2, -1)) / mm.sum(axis=(-2, -1))
            sd = np.sqrt((((v - mu[:, None, None]) ** 2) * mm).sum(axis=(-2, -1)) / mm.sum(axis=(-2, -1)))
            idx += [mu, sd]
    return np.concatenate([mean, std, np.stack(idx, axis=1)], axis=1).astype(np.float32)


def cell_spectral(s2, nobs):
    """(N, 12, 224, 224) + (N, 224, 224) -> (N, 49, 14): médias das bandas, NDVI e NBR em cada célula de 32 x 32 px."""
    g, c = config.CELL_GRID, config.PATCH_SIZE // config.CELL_GRID
    x = s2.astype(np.float32) / 10000
    feats = np.concatenate([x, np.stack(_ndvi_nbr(x), axis=1)], axis=1)          # (N, 14, H, W)
    n, f = feats.shape[:2]
    m = (nobs > 0).astype(np.float32).reshape(n, 1, g, c, g, c)
    s = (feats.reshape(n, f, g, c, g, c) * m).sum(axis=(3, 5))
    cnt = m.sum(axis=(3, 5))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = s / cnt                                                              # (N, 14, 7, 7)
    return out.reshape(n, f, g * g).transpose(0, 2, 1).astype(np.float16)


# ---------------- armazenamento ----------------

def empty_store(slot_pids, steps=config.STEPS):
    s, t = len(slot_pids), len(steps)
    return {'patch_id': np.asarray(slot_pids, dtype='<U32'), 'steps': np.asarray(steps),
            'z_s2': np.zeros((s, t, D), np.float16), 'z_s2s1': np.zeros((s, t, D), np.float16),
            'spec': np.full((s, t, len(SPEC_NAMES)), np.nan, np.float32), 'spec_names': np.asarray(SPEC_NAMES),
            'clear_s2': np.zeros((s, t), np.float16), 'clear_s1': np.zeros((s, t), np.float16),
            'done': np.zeros((s, t), bool)}


def save_store(store, path):
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **store)
    tmp.replace(path)              # escrita atômica: um corte no meio não corrompe o arquivo


def load_store(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def cells_path(path):
    return path.with_name(path.stem + '_cells.zarr')


def open_cells(path, n_slots, n_steps, mode='a'):
    """Zarr das células; chunk = 1 bloco (16 slots) x 1 trimestre."""
    import zarr
    root = zarr.open_group(str(cells_path(path)), mode=mode)
    for name in CONFIGS:
        if f'c_{name}' not in root:
            root.create_array(f'c_{name}', shape=(n_slots, n_steps, config.N_CELLS, D),
                              chunks=(NB, 1, config.N_CELLS, D), dtype='float16', fill_value=0)
    if 'spec_cells' not in root:
        root.create_array('spec_cells', shape=(n_slots, n_steps, config.N_CELLS, len(CELL_SPEC_NAMES)),
                          chunks=(NB, 1, config.N_CELLS, len(CELL_SPEC_NAMES)), dtype='float16', fill_value=np.nan)
        root.attrs['spec_names'] = CELL_SPEC_NAMES
    return root


def write_step(store, cells, sl, k, t, res, s2, n2, n1):
    """Grava um (bloco, trimestre): `sl` = fatia de 16 slots, `k` = posições ocupadas dentro do bloco."""
    idx = np.arange(sl.start, sl.stop)[k]
    for name in CONFIGS:
        z, c, _ = res[name]
        store[f'z_{name}'][idx, t] = z.astype(np.float16)
        full = np.zeros((NB, config.N_CELLS, D), np.float16)
        full[k] = c
        cells[f'c_{name}'][sl, t] = full
    sc = np.full((NB, config.N_CELLS, len(CELL_SPEC_NAMES)), np.nan, np.float16)
    sc[k] = cell_spectral(s2, n2)
    cells['spec_cells'][sl, t] = sc
    store['spec'][idx, t] = spectral_features(s2, n2)
    store['clear_s2'][idx, t] = (n2 > 0).mean(axis=(-2, -1))
    store['clear_s1'][idx, t] = (n1 > 0).mean(axis=(-2, -1))
    store['done'][idx, t] = True


# ---------------- sub-AOI: a partir do cubo ----------------

def extract_cube(cube_path, out_path, model=None, log_every=4):
    """Embeddings (S2 e S2 + S1), células, tokens e features espectrais de todos os passos do cubo. Retomável."""
    import zarr
    root = zarr.open_group(str(cube_path), mode='r')
    pids, steps = list(root.attrs['patch_id']), list(root.attrs['steps'])
    n, T = len(pids), len(steps)
    assert n % NB == 0, 'o cubo deve ter blocos completos de 16 patches'
    store = load_store(out_path) if out_path.exists() else empty_store(pids, steps)
    cells = open_cells(out_path, n, T)
    tok = zarr.open_group(str(out_path.with_name(out_path.stem + '_tokens.zarr')), mode='a')
    for name in CONFIGS:
        if f'tok_{name}' not in tok:
            tok.create_array(f'tok_{name}', shape=(n, T, 196, D), chunks=(16, 1, 196, D), dtype='float16')
    model = model or load_model()
    t0 = time.time()
    for t in range(T):
        if store['done'][:, t].all():
            continue
        s2, n2 = root['s2'][:, t], root['s2_nobs'][:, t]
        s1, n1 = root['s1'][:, t].astype(np.float32), root['s1_nobs'][:, t]
        res = embed_all_configs(model, s2, n2, s1, n1, tokens=True)
        for name in CONFIGS:
            tok[f'tok_{name}'][:, t] = res[name][2]
        for b in range(n // NB):
            sl = slice(b * NB, (b + 1) * NB)
            write_step(store, cells, sl, np.arange(NB), t, {k: tuple(a[sl] for a in v[:2]) + (None,) for k, v in res.items()},
                       s2[sl], n2[sl], n1[sl])
        save_store(store, out_path)
        if (t + 1) % log_every == 0:
            print(f'  passo {t + 1}/{T}  {(time.time() - t0) / 60:.1f} min', flush=True)
    return store


def cells_from_tokens(cube_path, out_path):
    """Para uma sub-AOI extraída antes de 29/10 (sem células): células a partir dos tokens já salvos, sem rodar o modelo."""
    import zarr
    store = load_store(out_path)
    n, T = store['done'].shape
    cells = open_cells(out_path, n, T)
    tok = zarr.open_group(str(out_path.with_name(out_path.stem + '_tokens.zarr')), mode='r')
    cube = zarr.open_group(str(cube_path), mode='r')
    for t in range(T):
        for name in CONFIGS:
            cells[f'c_{name}'][:, t] = pool_cells(tok[f'tok_{name}'][:, t]).astype(np.float16)
        cells['spec_cells'][:, t] = cell_spectral(cube['s2'][:, t], cube['s2_nobs'][:, t])
    return cells


# ---------------- AOIs completas: em fluxo, sem guardar pixels ----------------

def block_layout(grade):
    """Para cada bloco (ordem fixa): canto inferior esquerdo (UTM), linhas da grade e posição k (0–15) no bloco."""
    B, PM = config.BLOCK_PATCHES, config.PATCH_M
    g = grade.reset_index(drop=True).assign(i=lambda d: d.index)
    out = []
    for blk, b in g.groupby('block', sort=True):
        bc, br = int(b['col'].iloc[0]) // B, int(b['row'].iloc[0]) // B
        k = (B - 1 - (b['row'] - br * B)) * B + (b['col'] - bc * B)       # ordem de dados.block_to_patches
        out.append({'block': blk, 'x0': bc * B * PM, 'y0': br * B * PM, 'idx': b['i'].values, 'k': k.values})
    return out


def slot_patch_ids(grade, layout):
    pids = np.full(len(layout) * NB, '', dtype='<U32')
    for b, L in enumerate(layout):
        pids[b * NB + L['k']] = grade['patch_id'].values[L['idx']]
    return pids


def stream_aoi(aoi, out_path, workers=10, model=None, max_tasks=None, save_every_s=300, seed=config.SEED):
    """Baixa S2 + S1 de cada (bloco, trimestre), extrai embeddings de patch e de célula e descarta os pixels.

    Blocos em ordem aleatória (semente fixa): se o tempo acabar, o que já terminou é uma amostra
    aleatória de blocos, válida para métricas. Retomável: pula (bloco, trimestre) com `done`."""
    grade = pd.read_csv(config.LABELS_DIR / aoi / 'grade.csv')
    epsg = config.AOIS[aoi]['epsg']
    layout = block_layout(grade)
    pids = slot_patch_ids(grade, layout)
    store = load_store(out_path) if out_path.exists() else empty_store(pids)
    assert list(store['patch_id']) == list(pids), 'arquivo existente com outra grade'
    steps = list(store['steps'])
    cells = open_cells(out_path, len(pids), len(steps))
    occupied = pids != ''
    order = np.random.default_rng(seed).permutation(len(layout))
    todo = [(b, t) for b in order for t in range(len(steps))
            if not store['done'][b * NB:(b + 1) * NB][occupied[b * NB:(b + 1) * NB], t].all()]
    if max_tasks:
        todo = todo[:max_tasks]
    print(f'{aoi}: {len(todo)} de {len(layout) * len(steps)} (bloco, trimestre) pendentes; '
          f'{workers} downloads em paralelo', flush=True)

    def fetch(b, t):
        s, e = rotulos.step_bounds(steps[t])
        args = (layout[b]['x0'], layout[b]['y0'], epsg, s.strftime('%Y-%m-%d'),
                (e + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        return b, t, dados.fetch_block('s2', *args), dados.fetch_block('s1', *args)

    model = model or load_model()
    t0, last_save, n_done = time.time(), time.time(), 0
    it = iter(todo)
    with ThreadPoolExecutor(workers) as pool:
        running = {pool.submit(fetch, *x) for x in [next(it, None) for _ in range(2 * workers)] if x}
        while running:
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                b, t, (s2, n2), (s1, n1) = f.result()
                k = np.sort(layout[b]['k'])                      # posições ocupadas, em ordem
                s2, n2 = dados.block_to_patches(s2)[k], dados.block_to_patches(n2)[k]
                s1, n1 = dados.block_to_patches(s1)[k], dados.block_to_patches(n1)[k]
                res = embed_all_configs(model, s2, n2, s1, n1)
                write_step(store, cells, slice(b * NB, (b + 1) * NB), k, t, res, s2, n2, n1)
                n_done += 1
                nxt = next(it, None)
                if nxt:
                    running.add(pool.submit(fetch, *nxt))
            if time.time() - last_save > save_every_s or not running:
                save_store(store, out_path)
                last_save = time.time()
                el = time.time() - t0
                print(f'  {n_done}/{len(todo)} nesta sessão | {store["done"][occupied].mean():.1%} da AOI | '
                      f'{el / 3600:.1f} h | ~{el / n_done * (len(todo) - n_done) / 3600:.1f} h restantes', flush=True)
    return store


# ---------------- gate: pontos do first pass ----------------

def fetch_points(pairs, epsg, out_path, workers=10, model=None, chunk=64):
    """Composição ANUAL S2 (+S1) num patch 224 x 224 centrado em cada ponto, para cada par (ponto, ano).
    pairs: DataFrame com lon, lat, year (uma linha por par). Retomável. Guarda a média do patch e a célula
    central (que contém o ponto), para comparar as duas escalas."""
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = tr.transform(pairs['lon'].values, pairs['lat'].values)
    half = config.PATCH_M / 2
    x0 = (np.round((x - half) / config.RESOLUTION_M) * config.RESOLUTION_M).astype(int)
    y0 = (np.round((y - half) / config.RESOLUTION_M) * config.RESOLUTION_M).astype(int)
    yr = pairs['year'].values.astype(int)
    n = len(pairs)
    centre = (config.CELL_GRID // 2) * config.CELL_GRID + config.CELL_GRID // 2     # célula 24 de 49
    if out_path.exists():
        st = load_store(out_path)
        assert len(st['done']) == n, 'arquivo existente com outra lista de pares'
    else:
        st = {'clear_s2': np.zeros(n, np.float32), 'done': np.zeros(n, bool)}
        for name in CONFIGS:
            st[f'z_{name}'] = np.full((n, D), np.nan, np.float32)
            st[f'cell_{name}'] = np.full((n, D), np.nan, np.float32)
    todo = np.nonzero(~st['done'])[0]
    print(f'{len(todo)} de {n} (ponto, ano) pendentes', flush=True)

    def fetch(i):
        a = (int(x0[i]), int(y0[i]), epsg, f'{yr[i]}-01-01', f'{yr[i] + 1}-01-01')
        return (i, dados.fetch_block('s2', *a, size_px=config.PATCH_SIZE),
                dados.fetch_block('s1', *a, size_px=config.PATCH_SIZE))

    model = model or load_model()
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for c0 in range(0, len(todo), chunk):            # lotes: limita a memória e salva a cada lote
            got = list(pool.map(fetch, todo[c0:c0 + chunk]))
            ii = [g[0] for g in got]
            s2 = np.stack([g[1][0] for g in got]); n2 = np.stack([g[1][1] for g in got])
            s1 = np.stack([g[2][0] for g in got]); n1 = np.stack([g[2][1] for g in got])
            res = embed_all_configs(model, s2, n2, s1, n1)
            for name in CONFIGS:
                st[f'z_{name}'][ii] = res[name][0]
                st[f'cell_{name}'][ii] = res[name][1][:, centre].astype(np.float32)
            st['clear_s2'][ii] = (n2 > 0).mean(axis=(-2, -1))
            st['done'][ii] = True
            save_store(st, out_path)
            k = min(c0 + chunk, len(todo)); el = time.time() - t0
            print(f'  {k}/{len(todo)}  {el / 60:.1f} min  (~{el / k * (len(todo) - k) / 60:.0f} min restantes)', flush=True)
    return st
