"""Etapa 5: baselines de detecção de anomalia na CÉLULA de 320 m, agregados no patch pelo máximo.

Uma amostra de avaliação = (patch, trimestre t) com pelo menos K_HIST trimestres anteriores e S2
observado em t. Cada método pontua as 49 células do patch; o score do patch é o máximo (e, para
comparar escalas, também a média). Score maior = mais anômalo.

Ajuste só com normais (regra por janela, PROTOCOLO.md): até N_FIT células sorteadas de amostras
normais de `train` (T <= 2022), e N_VAL de `val` para a parada antecipada do Deep SVDD.
Hiperparâmetros fixados a priori (k = 10, 300 árvores). Nenhum rótulo de anomalia entra no ajuste.
Duas passadas em lotes de patches, para caber na memória nas AOIs completas (~11 milhões de células).
"""
import time

import numpy as np
import pandas as pd

import config
import embeddings
import rotulos

N_FIT, N_VAL = 40_000, 10_000
NC = config.N_CELLS


# ---------------- dados ----------------

def history_mean(X, valid, k=config.K_HIST, min_valid=None):
    """X (..., T, F) com valid (..., T) -> média dos k passos anteriores válidos; NaN se < min_valid."""
    min_valid = min_valid or max(1, k // 2)
    T = X.shape[-2]
    H = np.full(X.shape, np.nan, np.float32)
    Xz = np.nan_to_num(X.astype(np.float32))
    for t in range(k, T):
        w = valid[..., t - k:t].astype(np.float32)
        c = w.sum(-1)
        s = (Xz[..., t - k:t, :] * w[..., None]).sum(-2)
        ok = c >= min_valid
        H[..., t, :][ok] = s[ok] / c[ok][..., None]
    return H


def load(name, aoi, k=config.K_HIST):
    """Tabela de amostras (patch, trimestre) e o contexto para ler as células em lotes."""
    import zarr
    path = config.EMBEDDINGS_DIR / f'{name}.npz'
    st = embeddings.load_store(path)
    cells = zarr.open_group(str(embeddings.cells_path(path)), mode='r')
    d = config.LABELS_DIR / aoi
    grade = pd.read_csv(d / 'grade.csv').set_index('patch_id')
    anual = pd.read_csv(d / 'rotulos_anuais.csv.gz')
    trim = pd.read_csv(d / 'rotulos_trimestrais.csv.gz')
    pids, steps = st['patch_id'], np.asarray(st['steps'])
    valid = st['done'] & (st['clear_s2'].astype(np.float32) >= config.MIN_CLEAR_FRAC) & (pids != '')[:, None]

    # histórico suficiente: >= k/2 passos válidos entre os k anteriores
    hist_ok = np.zeros_like(valid)
    for t in range(k, valid.shape[1]):
        hist_ok[:, t] = valid[:, t - k:t].sum(1) >= max(1, k // 2)
    ii, tt = np.nonzero(valid & hist_ok)
    s = pd.DataFrame({'i': ii, 't': tt, 'patch_id': pids[ii], 'step': steps[tt]})
    s['year'] = s['step'].str[:4].astype(int)
    g = grade.reindex(s['patch_id'])
    for c in ['split', 'block', 'cobertura']:
        s[c] = g[c].values
    nw = rotulos.normal_windows(anual).set_index(['patch_id', 'year'])['normal_janela']
    s['normal_janela'] = nw.reindex(pd.MultiIndex.from_arrays([s['patch_id'], s['year']])).fillna(False).values
    s['treino_normal'] = (s['split'] == 'train') & (s['year'] <= config.TRAIN_END_YEAR) & s['normal_janela']
    s['val_normal'] = (s['split'] == 'val') & (s['year'] <= config.TRAIN_END_YEAR) & s['normal_janela']
    ctx = {'cells': cells, 'valid': valid, 'k': k, 'n_slots': len(pids)}
    return s, ctx, {'anual': anual, 'trim': trim, 'grade': grade.reset_index()}


def _batches(s, ctx, batch_slots=64):
    """Lê as células em lotes de slots e devolve, para as amostras do lote, as features por célula:
    Z (n, 49, D) e delta (n, 49, D) por configuração, e espectral (n, 49, 28) = [F, F - histórico]."""
    cells, valid, k = ctx['cells'], ctx['valid'], ctx['k']
    for b0 in range(0, ctx['n_slots'], batch_slots):
        rows = s[(s['i'] >= b0) & (s['i'] < b0 + batch_slots)]
        if rows.empty:
            continue
        sl = slice(b0, b0 + batch_slots)
        v = valid[sl]                                                    # (b, T)
        vc = np.repeat(v[:, None, :], NC, axis=1)                         # (b, 49, T)
        li, tt = rows['i'].values - b0, rows['t'].values
        out = {}
        for cfg in embeddings.CONFIGS:
            Z = cells[f'c_{cfg}'][sl].astype(np.float32).transpose(0, 2, 1, 3)   # (b, 49, T, D)
            H = history_mean(Z, vc, k)
            out[f'Z_{cfg}'] = Z[li, :, tt]                               # (n, 49, D)
            out[f'H_{cfg}'] = H[li, :, tt]
            out[f'Zall_{cfg}'] = (Z, li, tt)                             # para o cosseno sazonal
        F = cells['spec_cells'][sl].astype(np.float32).transpose(0, 2, 1, 3)    # (b, 49, T, 14)
        HF = history_mean(F, vc & ~np.isnan(F).any(-1), k)
        f, hf = F[li, :, tt], HF[li, :, tt]
        out['spec'] = np.nan_to_num(np.concatenate([f, f - hf], axis=-1))
        out['ndvi_drop'] = np.nan_to_num(hf[..., 12] - f[..., 12])
        out['vt'] = (v, li, tt)
        yield rows, out


def _cos(a, b):
    return 1 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)


def _seasonal(out, cfg, lags=(4, 8)):
    """1 − cos entre a célula em t e a média da mesma célula em t-4 e t-8 (mesma estação). NaN se faltar."""
    Z, li, tt = out[f'Zall_{cfg}']
    v = out['vt'][0]
    ok = np.all([(tt - l >= 0) for l in lags], axis=0)
    tl = [np.maximum(tt - l, 0) for l in lags]
    ok &= np.all([v[li, x] for x in tl], axis=0)
    h = np.mean([Z[li, :, x] for x in tl], axis=0)
    c = _cos(Z[li, :, tt], h)
    c[~ok] = np.nan
    return c


# ---------------- detectores ----------------

class Standardize:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6

    def __call__(self, X):
        return (X - self.mu) / self.sd


class KNN:
    """Distância média aos k vizinhos normais de treino (cosseno para Z estático; euclidiana padronizada no delta)."""
    def __init__(self, Xtr, k=10, metric='euclidean'):
        from sklearn.neighbors import NearestNeighbors
        self.metric, self.k = metric, k
        self.z = None if metric == 'cosine' else Standardize(Xtr)
        self.nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(self._prep(Xtr))

    def _prep(self, X):
        return X / np.linalg.norm(X, axis=1, keepdims=True) if self.metric == 'cosine' else self.z(X)

    def __call__(self, X):
        return self.nn.kneighbors(self._prep(X))[0].mean(1)


class IForest:
    def __init__(self, Xtr, seed=config.SEED):
        from sklearn.ensemble import IsolationForest
        self.z = Standardize(Xtr)
        self.m = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(self.z(Xtr))

    def __call__(self, X):
        return -self.m.score_samples(self.z(X))


class DeepSVDD:
    """Deep SVDD de uma classe (Ruff et al., 2018): rede sem bias, centro fixo, parada pela perda na validação."""
    def __init__(self, Xtr, Xva, epochs=200, patience=15, lr=1e-3, batch=512, seed=config.SEED):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.z = Standardize(Xtr)
        tr, va = (torch.tensor(self.z(a), dtype=torch.float32) for a in (Xtr, Xva))
        self.net = nn.Sequential(nn.Linear(tr.shape[1], 256, bias=False), nn.LeakyReLU(),
                                 nn.Linear(256, 64, bias=False))
        with torch.no_grad():
            c = self.net(tr).mean(0)
            c[c.abs() < 0.01] = 0.01                     # evita centro em zero (solução trivial)
        self.c = c
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        best, best_state, wait, self.hist = np.inf, None, 0, []
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            self.net.train()
            for idx in torch.randperm(len(tr), generator=g).split(batch):
                loss = ((self.net(tr[idx]) - c) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = ((self.net(va) - c) ** 2).sum(1).mean().item()
            self.hist.append(v)
            if v < best - 1e-6:
                best, best_state, wait = v, {k: w.clone() for k, w in self.net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.net.load_state_dict(best_state)

    def __call__(self, X):
        import torch
        with torch.no_grad():
            return ((self.net(torch.tensor(self.z(X), dtype=torch.float32)) - self.c) ** 2).sum(1).numpy()


# ---------------- tabela de baselines ----------------

def collect_normals(s, ctx, cfgs, seed=config.SEED):
    """Passada 1: sorteia células de amostras normais (treino e validação) para o ajuste."""
    rng = np.random.default_rng(seed)
    p_tr = min(1.0, N_FIT / max(1, s['treino_normal'].sum() * NC))
    p_va = min(1.0, N_VAL / max(1, s['val_normal'].sum() * NC))
    keys = [f'{f}_{c}' for c in cfgs for f in ('static', 'delta')] + ['spec']
    acc = {sp: {k: [] for k in keys} for sp in ('tr', 'va')}
    for rows, out in _batches(s, ctx):
        for sp, col, p in (('tr', 'treino_normal', p_tr), ('va', 'val_normal', p_va)):
            m = rows[col].values
            if not m.any():
                continue
            pick = rng.random((m.sum(), NC)) < p
            for c in cfgs:
                acc[sp][f'static_{c}'].append(out[f'Z_{c}'][m][pick])
                acc[sp][f'delta_{c}'].append((out[f'Z_{c}'] - out[f'H_{c}'])[m][pick])
            acc[sp]['spec'].append(out['spec'][m][pick])
    return {sp: {k: np.concatenate(v) for k, v in d.items()} for sp, d in acc.items()}


def fit_all(fitdata, cfgs):
    tr, va = fitdata['tr'], fitdata['va']
    models = {}
    for c in cfgs:
        tag = f'TerraMind {c.upper()}'
        models[f'{tag} | kNN estático (sem tempo)'] = ('static', c, KNN(tr[f'static_{c}'], metric='cosine'))
        models[f'{tag} | kNN no delta'] = ('delta', c, KNN(tr[f'delta_{c}']))
        models[f'{tag} | IForest no delta'] = ('delta', c, IForest(tr[f'delta_{c}']))
        models[f'{tag} | Deep SVDD no delta'] = ('delta', c, DeepSVDD(tr[f'delta_{c}'], va[f'delta_{c}']))
    models['Espectral | IForest (bandas + delta)'] = ('spec', None, IForest(tr['spec']))
    return models


def score_all(s, ctx, models, cfgs, log=True):
    """Passada 2: pontua todas as células e agrega por amostra: máximo (principal) e média das 49 células."""
    parts, t0, done = [], time.time(), 0
    for rows, out in _batches(s, ctx):
        n = len(rows)
        cell = {}
        for c in cfgs:
            tag = f'TerraMind {c.upper()}'
            cell[f'{tag} | cosseno vs histórico'] = _cos(out[f'Z_{c}'], out[f'H_{c}'])
            cell[f'{tag} | cosseno vs mesmo trimestre (t-4, t-8)'] = _seasonal(out, c)
        cell['Espectral | queda de NDVI'] = out['ndvi_drop']
        for name, (kind, c, m) in models.items():
            X = out[f'Z_{c}'] if kind == 'static' else (out[f'Z_{c}'] - out[f'H_{c}'] if kind == 'delta' else out['spec'])
            cell[name] = m(X.reshape(n * NC, -1)).reshape(n, NC)
        agg = {}
        for name, v in cell.items():
            with np.errstate(all='ignore'):
                agg[name] = np.nanmax(v, axis=1) if not np.isnan(v).all() else np.full(n, np.nan)
                agg[f'{name} [média]'] = np.nanmean(v, axis=1)
        parts.append(pd.DataFrame(agg, index=rows.index))
        done += n
        if log:
            print(f'  {done}/{len(s)} amostras  {(time.time() - t0) / 60:.1f} min', flush=True)
    return pd.concat(parts).reindex(s.index)


def run_all(s, ctx, cfgs=('s2s1', 's2')):
    assert s['treino_normal'].any() and s['val_normal'].any(), 'sem normais de treino ou validação'
    t0 = time.time()
    fitdata = collect_normals(s, ctx, cfgs)
    print(f'células de ajuste: treino {len(fitdata["tr"]["spec"])}, val {len(fitdata["va"]["spec"])} '
          f'({(time.time() - t0) / 60:.1f} min)', flush=True)
    models = fit_all(fitdata, cfgs)
    info = {f'svdd_{c}': models[f'TerraMind {c.upper()} | Deep SVDD no delta'][2].hist for c in cfgs}
    print(f'ajuste feito ({(time.time() - t0) / 60:.1f} min); pontuando', flush=True)
    return score_all(s, ctx, models, cfgs), info


def to_annual(s, scores):
    """Score anual = máximo dos trimestres do ano (para comparar com rótulos anuais e com o AlphaEarth)."""
    return pd.concat([s[['patch_id', 'year']], scores], axis=1).groupby(['patch_id', 'year']).max().reset_index()


# ---------------- AlphaEarth nas mesmas células de 320 m ----------------

def alphaearth_cells(grid, name, cache_dir, years=config.YEARS, chunk=250, workers=8):
    """Embedding AlphaEarth médio de cada célula de 320 m (64-d por ano), via Earth Engine. Retomável (cache por lote).
    Retorna patch_ids (N,) e Z (N, 49, anos, 64)."""
    import ee
    from concurrent.futures import ThreadPoolExecutor
    from common import ae_image, AE_BANDS
    import dados
    dados._ensure_deadline()
    img = ee.Image.cat([ae_image(y).rename([f'{b}_{y}' for b in AE_BANDS]) for y in years])
    crs, G, CM = grid.crs.to_string(), config.CELL_GRID, config.CELL_M
    recs = [(r.patch_id, j, r.x0 + (j % G) * CM, r.y0 + config.PATCH_M - (j // G + 1) * CM)
            for r in grid.itertuples() for j in range(NC)]
    parts = cache_dir / 'ee_parts' / f'alphaearth_cells_{name}'
    parts.mkdir(parents=True, exist_ok=True)

    def job(c0):
        pf = parts / f'{c0 // chunk:05d}.csv'
        if pf.exists():
            return
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([x, y, x + CM, y + CM], crs, False),
                                              {'patch_id': p, 'cell': j}) for p, j, x, y in recs[c0:c0 + chunk]])
        for attempt in range(5):
            try:
                res = img.reduceRegions(fc, ee.Reducer.mean(), scale=30, crs=crs, tileScale=4).getInfo()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(15 * (attempt + 1))
        pd.DataFrame([f['properties'] for f in res['features']]).to_csv(pf, index=False)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(job, range(0, len(recs), chunk)))
    df = pd.concat([pd.read_csv(p) for p in sorted(parts.glob('*.csv'))], ignore_index=True)
    pids = grid['patch_id'].values
    df = df.set_index(['patch_id', 'cell']).reindex(pd.MultiIndex.from_product([pids, range(NC)]))
    Z = np.stack([df[[f'{b}_{y}' for b in AE_BANDS]].values for y in years], axis=1)
    return pids, Z.reshape(len(pids), NC, len(years), 64).astype(np.float32)


def alphaearth_scores(pids, Z, anual, grade, years=config.YEARS, hist=2, seed=config.SEED):
    """Mesmos scores do first pass, por célula e ano, agregados no patch pelo máximo (e pela média):
    cosseno vs média de T-2, T-1, e kNN estático ajustado em células normais de treino."""
    g = grade.set_index('patch_id').reindex(pids)
    nw = rotulos.normal_windows(anual).set_index(['patch_id', 'year'])['normal_janela']
    rows, Xtr = [], []
    rng = np.random.default_rng(seed)
    for j in range(hist, len(years)):
        a, h = Z[:, :, j], Z[:, :, j - hist:j].mean(2)                  # (N, 49, 64)
        normal = nw.reindex(pd.MultiIndex.from_arrays([pids, np.full(len(pids), years[j])])).fillna(False).values
        tr = normal & (g['split'].values == 'train') & (years[j] <= config.TRAIN_END_YEAR)
        if tr.any():
            Xtr.append(a[tr].reshape(-1, 64))
        rows.append((j, a, _cos(a, h)))
    Xtr = np.concatenate(Xtr)
    Xtr = Xtr[rng.choice(len(Xtr), min(N_FIT, len(Xtr)), replace=False)]
    knn = KNN(Xtr, metric='cosine')
    out = []
    for j, a, cos in rows:
        k = knn(a.reshape(-1, 64)).reshape(a.shape[:2])
        d = {'patch_id': pids, 'year': years[j]}
        for name, v in [('AlphaEarth | cosseno vs histórico', cos), ('AlphaEarth | kNN estático (sem tempo)', k)]:
            d[name], d[f'{name} [média]'] = np.nanmax(v, axis=1), np.nanmean(v, axis=1)
        out.append(pd.DataFrame(d))
    return pd.concat(out, ignore_index=True)
