"""Etapa 2: grade de patches, rótulos de referência (só avaliação; o modelo não os vê) e divisão espacial.

Fontes vetoriais (PRODES, DETER) vêm do WFS do TerraBrasilis e são cruzadas com a grade em geopandas.
Fontes raster (Hansen, MCD64A1, MapBiomas) são reduzidas por patch no Earth Engine.
Tudo é guardado em formato longo (`eventos`) e depois agregado em tabelas anuais e trimestrais.
"""
import re
import time

import numpy as np
import pandas as pd

import config


# ---------------- tempo ----------------

def step_of(dates):
    d = pd.to_datetime(dates)
    return d.dt.year.astype(str) + 'Q' + ((d.dt.month - 1) // 3 + 1).astype(str)


def step_bounds(step):
    y, q = int(step[:4]), int(step[-1])
    start = pd.Timestamp(y, 3 * q - 2, 1)
    return start, start + pd.offsets.QuarterEnd(0)


# ---------------- grade ----------------

def make_grid(aoi):
    """Patches de PATCH_M x PATCH_M em UTM, alinhados a múltiplos de PATCH_M (reaproveitados na Etapa 3)."""
    import geopandas as gpd
    from shapely.geometry import box

    a, P, B = config.AOIS[aoi], config.PATCH_M, config.BLOCK_PATCHES
    aoi_utm = gpd.GeoSeries([box(*a['bounds'])], crs=4326).to_crs(a['epsg'])
    x0, y0, x1, y1 = aoi_utm.total_bounds
    cols = np.arange(np.floor(x0 / P), np.ceil(x1 / P)).astype(int)
    rows = np.arange(np.floor(y0 / P), np.ceil(y1 / P)).astype(int)
    C, R = (m.ravel() for m in np.meshgrid(cols, rows))
    g = gpd.GeoDataFrame({'col': C, 'row': R, 'x0': C * P, 'y0': R * P},
                         geometry=[box(c * P, r * P, (c + 1) * P, (r + 1) * P) for c, r in zip(C, R)],
                         crs=a['epsg'])
    cen = g.geometry.centroid.to_crs(4326)
    lo0, la0, lo1, la1 = a['bounds']
    g = g[(cen.x >= lo0) & (cen.x < lo1) & (cen.y >= la0) & (cen.y < la1)].copy()
    g['lon'], g['lat'] = cen.x[g.index].round(5), cen.y[g.index].round(5)
    g.insert(0, 'patch_id', [f'{aoi}_{c}_{r}' for c, r in zip(g.col, g.row)])
    g['aoi'] = aoi
    g['block'] = [f'{aoi}_{c // B}_{r // B}' for c, r in zip(g.col, g.row)]
    return g.reset_index(drop=True)


def assign_blocks_to_splits(df, seed=config.SEED, target=config.SPLIT_TARGET):
    """Aloca blocos inteiros a train/val/test (mesmo algoritmo guloso de notebooks/01_dados.ipynb)."""
    rng = np.random.default_rng(seed)
    block_ids = np.array(df['block'].unique())
    rng.shuffle(block_ids)
    names = ['train', 'val', 'test']
    counts = {n: 0 for n in names}
    sizes = df.groupby('block').size().to_dict()
    assignment = {}
    for bid in block_ids:
        total = sum(counts.values()) or 1
        deficits = {n: target[i] - counts[n] / total for i, n in enumerate(names)}
        chosen = max(deficits, key=deficits.get)
        assignment[bid] = chosen
        counts[chosen] += sizes[bid]
    return df['block'].map(assignment)


# ---------------- fontes vetoriais (TerraBrasilis WFS) ----------------

def fetch_wfs(layer, bounds, cql, sort_by, cache, page=5000, retries=4):
    """Baixa um layer WFS paginado dentro de `bounds` (lon/lat) e guarda em cache (GeoPackage)."""
    import geopandas as gpd
    import requests

    if cache.exists():
        return gpd.read_file(cache)
    ws, name = layer
    flt = f"BBOX(geom,{','.join(map(str, bounds))},'EPSG:4326') AND {cql}"
    feats, start = [], 0
    while True:
        params = dict(service='WFS', version='2.0.0', request='GetFeature', typeNames=f'{ws}:{name}',
                      CQL_FILTER=flt, count=page, startIndex=start, sortBy=sort_by,
                      outputFormat='application/json', srsName='EPSG:4326')
        for attempt in range(retries):
            try:
                r = requests.get(f'{config.TERRABRASILIS_WFS}/{ws}/ows', params=params, timeout=600)
                r.raise_for_status()
                page_feats = r.json()['features']
                break
            except Exception as e:
                print('  retry', attempt + 1, str(e)[:120], flush=True)
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError(f'WFS {name} falhou')
        feats += page_feats
        print(f'  {name}: {len(feats)} feições', flush=True)
        if len(page_feats) < page:
            break
        start += page
    gdf = gpd.GeoDataFrame.from_features(feats, crs=4326)
    cache.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache, driver='GPKG')
    return gdf


def fetch_prodes(aoi, cache_dir):
    y0, y1 = config.YEARS[0], config.YEARS[-1]
    g = fetch_wfs(config.PRODES_LAYER, config.AOIS[aoi]['bounds'], f'year>={y0} AND year<={y1}',
                  'uuid', cache_dir / 'prodes.gpkg')
    # ano PRODES Y = ago/(Y-1) a jul/Y; o fim da janela é a data da imagem em que o corte foi mapeado
    start = pd.to_datetime((g['year'] - 1).astype(str) + '-08-01')
    end = pd.to_datetime((g['year']).astype(str) + '-07-31')
    img = pd.to_datetime(g['image_date'], errors='coerce')
    img = img.where((img >= start) & (img <= end), end)
    return g.assign(event_id='prodes_' + g['uuid'].astype(str), source='prodes', tipo='desmatamento',
                    date_start=start, date_end=img)


def fetch_deter(aoi, cache_dir):
    y0, y1 = config.YEARS[0], config.YEARS[-1]
    g = fetch_wfs(config.DETER_LAYER, config.AOIS[aoi]['bounds'],
                  f"view_date>='{y0}-01-01' AND view_date<='{y1}-12-31'", 'gid', cache_dir / 'deter.gpkg')
    g = g[g['classname'].isin(config.DETER_CLASSES)].copy()
    d = pd.to_datetime(g['view_date'])
    return g.assign(event_id='deter_' + g['gid'].astype(str), source='deter',
                    tipo=g['classname'].map(config.DETER_CLASSES), date_start=d, date_end=d)


def overlay_events(grid, polys):
    """Fração de cada patch coberta por cada polígono de evento -> linhas do formato longo."""
    import geopandas as gpd

    cols = ['event_id', 'source', 'tipo', 'date_start', 'date_end']
    p = polys[cols + ['geometry']].to_crs(grid.crs)
    # algumas feições do DETER chegam como GeometryCollection, e make_valid também pode gerá-las:
    # explode antes e depois, ficando só com as partes poligonais
    poly = ['Polygon', 'MultiPolygon']
    p = p.explode(index_parts=False)
    p = p[p.geom_type.isin(poly)]
    p = p.set_geometry(p.geometry.make_valid()).explode(index_parts=False)
    p = p[p.geom_type.isin(poly)]
    inter = gpd.overlay(grid[['patch_id', 'geometry']], p, how='intersection', keep_geom_type=True)
    inter['frac'] = inter.geometry.area / config.PATCH_M ** 2
    return (pd.DataFrame(inter.drop(columns='geometry'))
            .groupby(['patch_id', *cols], as_index=False)['frac'].sum())


def event_table(polys):
    """Uma linha por polígono de evento (para métricas por evento e antecedência)."""
    return pd.DataFrame({'event_id': polys['event_id'], 'source': polys['source'], 'tipo': polys['tipo'],
                         'date_start': polys['date_start'], 'date_end': polys['date_end'],
                         'area_ha': polys.to_crs(polys.estimate_utm_crs()).geometry.area / 1e4})


# ---------------- fontes raster (Earth Engine) ----------------

def _lulc_tools():
    from common import lulc, isin  # first pass
    return lulc, isin


def hansen_image():
    import ee
    for asset in config.HANSEN_ASSETS:
        try:
            img = ee.Image(asset)
            img.bandNames().getInfo()
            break
        except ee.EEException:
            continue
    else:
        raise RuntimeError('nenhum asset Hansen disponível')
    last = int(re.search(r'change_(\d{4})_', asset).group(1))
    ly = img.select('lossyear').unmask(0)
    bands = [ly.eq(y - 2000).rename(f'hansen__desmatamento__{y}') for y in config.YEARS if y <= last]
    return ee.Image.cat(bands), {'asset': asset, 'last_year': last}


def mapbiomas_image():
    """Transições anuais (Y-1 -> Y) e composição de cobertura no ano-base."""
    import ee
    lulc, isin = _lulc_tools()
    avail = sorted(set(ee.ImageCollection(config.LULC_ASSET).aggregate_array('year').getInfo()))
    bands = []
    for y in config.YEARS:
        if y not in avail or y - 1 not in avail:
            continue
        a, b = lulc(y - 1), lulc(y)
        desm = (isin(a, config.NATURAL_FOREST).And(isin(b, config.ANTHROPIC))
                .And(isin(b, config.MINING).Not()))
        mine = isin(a, config.MINING).Not().And(isin(b, config.MINING))
        agua = isin(a, config.WATER).neq(isin(b, config.WATER))
        bands += [desm.rename(f'mapbiomas__desmatamento__{y}'), mine.rename(f'mapbiomas__mineracao__{y}'),
                  agua.rename(f'mapbiomas__agua__{y}')]
    base = lulc(config.YEARS[0])
    bands += [isin(base, codes).rename(f'cover__{g}') for g, codes in config.COVER_GROUPS.items()]
    return ee.Image.cat(bands).unmask(0), {'years_available': [y for y in avail if y >= config.YEARS[0] - 1]}


def mcd64_image():
    """Área queimada (MCD64A1) por trimestre, só sobre floresta natural do MapBiomas no ano anterior."""
    import ee
    lulc, isin = _lulc_tools()
    col = ee.ImageCollection(config.MCD64_ASSET)
    last = pd.Timestamp(ee.Date(col.aggregate_max('system:time_start')).format('YYYY-MM-dd').getInfo())
    lulc_years = set(ee.ImageCollection(config.LULC_ASSET).aggregate_array('year').getInfo())
    bands = []
    for step in config.STEPS:
        start, end = step_bounds(step)
        if start > last:
            continue
        fy = min(start.year - 1, max(lulc_years))
        burned = (col.filterDate(start.strftime('%Y-%m-%d'), (end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
                  .select('BurnDate').max().gt(0).unmask(0))
        bands.append(burned.And(isin(lulc(fy), config.NATURAL_FOREST)).rename(f'mcd64__fogo__{step}'))
    return ee.Image.cat(bands), {'last_image': str(last.date())}


def ee_reduce(image, grid, name, cache_dir, scale=30, chunk=400, retries=4):
    """Média de cada banda por patch (= fração do patch), em lotes com cache por lote (retomável)."""
    import ee

    out = cache_dir / f'ee_{name}.csv'
    if out.exists():
        return pd.read_csv(out)
    parts = cache_dir / 'ee_parts' / name
    parts.mkdir(parents=True, exist_ok=True)
    crs = grid.crs.to_string()
    for i in range(0, len(grid), chunk):
        pf = parts / f'{i // chunk:04d}.csv'
        if pf.exists():
            continue
        part = grid.iloc[i:i + chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Rectangle([r.x0, r.y0, r.x0 + config.PATCH_M, r.y0 + config.PATCH_M],
                                             crs, False), {'patch_id': r.patch_id})
            for r in part.itertuples()])
        for attempt in range(retries):
            try:
                res = image.reduceRegions(fc, ee.Reducer.mean(), scale=scale, crs=crs, tileScale=4).getInfo()
                break
            except Exception as e:
                print('  retry', attempt + 1, str(e)[:120], flush=True)
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError(f'reduceRegions {name} falhou no lote {i // chunk}')
        pd.DataFrame([f['properties'] for f in res['features']]).to_csv(pf, index=False)
        print(f'  {name}: {min(i + chunk, len(grid))}/{len(grid)} patches', flush=True)
    df = pd.concat([pd.read_csv(p) for p in sorted(parts.glob('*.csv'))], ignore_index=True)
    df.to_csv(out, index=False)
    return df


def raster_events(wide):
    """Colunas `fonte__tipo__periodo` (fração por patch) -> linhas do formato longo, só frações > 0."""
    cols = [c for c in wide.columns if c.count('__') == 2]
    long = wide.melt(id_vars='patch_id', value_vars=cols, var_name='band', value_name='frac')
    long = long[long['frac'] > 0].copy()
    long[['source', 'tipo', 'period']] = long['band'].str.split('__', expand=True)
    per = long['period']
    is_q = per.str.contains('Q').values
    pq = pd.PeriodIndex(np.where(is_q, per, per + 'Q1'), freq='Q')   # anual -> começa no 1º trimestre
    long['date_start'] = pq.start_time
    long['date_end'] = np.where(is_q, pq.end_time.normalize(), pd.to_datetime(per.str[:4] + '-12-31'))
    long['event_id'] = ''
    return long[['patch_id', 'source', 'tipo', 'event_id', 'date_start', 'date_end', 'frac']]


# ---------------- agregação ----------------

def _aggregate(ev, key, all_keys, patch_ids):
    """Soma por fonte (clip 1) e máximo entre fontes, para cada tipo; preenche zeros e classifica o estado."""
    per_src = ev.groupby(['patch_id', key, 'tipo', 'source'])['frac'].sum().clip(upper=1)
    by_type = per_src.groupby(['patch_id', key, 'tipo']).max().unstack('tipo')
    by_type = by_type.reindex(columns=config.EVENT_TYPES).add_prefix('frac_')
    idx = pd.MultiIndex.from_product([patch_ids, all_keys], names=['patch_id', key])
    out = by_type.reindex(idx).fillna(0.0).rename_axis(columns=None).reset_index()
    m = out.filter(like='frac_').max(axis=1)
    out['estado'] = np.select([m >= config.EVENT_MIN_FRAC, m > 0], ['evento', 'ambiguo'], 'limpo')
    return out


def annual_labels(eventos, patch_ids):
    ev = eventos.assign(year=pd.to_datetime(eventos['date_end']).dt.year)
    ev = ev[ev['year'].isin(config.YEARS)]
    out = _aggregate(ev, 'year', config.YEARS, patch_ids)
    out['periodo'] = np.where(out['year'] <= config.TRAIN_END_YEAR, 'treino', 'teste')
    return out


def quarterly_labels(eventos, patch_ids):
    """Só fontes com data sub-anual: DETER (data da imagem) e MCD64A1 (trimestre)."""
    ev = eventos[eventos['source'].isin(['deter', 'mcd64'])]
    ev = ev.assign(step=step_of(ev['date_end']))
    ev = ev[ev['step'].isin(config.STEPS)]
    out = _aggregate(ev, 'step', config.STEPS, patch_ids)
    out['periodo'] = np.where(out['step'].str[:4].astype(int) <= config.TRAIN_END_YEAR, 'treino', 'teste')
    return out


def normal_flags(eventos, patch_ids, tol=config.NORMAL_MAX_FRAC):
    """Normal de treino: em cada fonte, a fração somada em 2019–2025 fica abaixo de `tol`."""
    tot = eventos.groupby(['patch_id', 'source'])['frac'].sum().unstack('source')
    tot = tot.reindex(patch_ids).fillna(0.0)
    return pd.DataFrame({'patch_id': patch_ids,
                         'normal_treino': (tot.max(axis=1) < tol).values,
                         'normal_estrito': (tot.max(axis=1) == 0).values,
                         **{f'total_{s}': tot[s].values for s in tot.columns}})
