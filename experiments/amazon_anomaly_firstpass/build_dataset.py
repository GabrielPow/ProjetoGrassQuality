"""Sample labeled points (MapBiomas transitions) and extract AlphaEarth + Sentinel-2 per year.

Output: data/amazon_firstpass.npz (cached; delete it to rebuild).
"""
import os
import time
import numpy as np
import pandas as pd
from common import *

N_PER_STRATUM = 150
GRID_DEG = 0.1          # ~11 km blocks for the spatial train/test split
YEARS = list(range(min(EVENT_YEARS) - HIST, max(EVENT_YEARS) + 2))
OUT = 'data/amazon_firstpass.npz'


def chunks(df, n=300):
    for i in range(0, len(df), n):
        yield df.iloc[i:i + n]


def reduce_at_points(image, df, scale, retries=4):
    out = {}
    for part in chunks(df):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {'pid': int(r.pid)})
                                   for r in part.itertuples()])
        for attempt in range(retries):
            try:
                res = image.reduceRegions(fc, ee.Reducer.first(), scale=scale, tileScale=4).getInfo()
                break
            except Exception as e:
                print('  retry', attempt + 1, str(e)[:120], flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError('reduceRegions failed')
        for f in res['features']:
            out[f['properties']['pid']] = f['properties']
    return out


def main():
    init()
    os.makedirs('data', exist_ok=True)
    rows = []
    for T in EVENT_YEARS:
        fc = strata_image(T).stratifiedSample(
            numPoints=0, classBand='strat', region=aoi(), scale=30, seed=SEED + T,
            classValues=list(STRATA), classPoints=[N_PER_STRATUM] * len(STRATA),
            geometries=True, tileScale=8)
        feats = fc.getInfo()['features']
        for f in feats:
            lon, lat = f['geometry']['coordinates']
            rows.append(dict(lon=lon, lat=lat, T=T, strat=int(f['properties']['strat'])))
        print(f'T={T}: {len(feats)} points', flush=True)
    df = pd.DataFrame(rows).drop_duplicates(['lon', 'lat', 'T']).reset_index(drop=True)
    df['pid'] = np.arange(len(df))
    print(df.groupby(['T', 'strat']).size().unstack())

    n, ny = len(df), len(YEARS)
    AE = np.full((n, ny, 64), np.nan, np.float32)
    s2_names = S2_BANDS + ['NDVI', 'NBR']
    S2 = np.full((n, ny, len(s2_names)), np.nan, np.float32)
    for j, y in enumerate(YEARS):
        t = time.time()
        for pid, p in reduce_at_points(ae_image(y), df, scale=10).items():
            v = [p.get(b) for b in AE_BANDS]
            if all(x is not None for x in v):
                AE[pid, j] = v
        for pid, p in reduce_at_points(s2_image(y), df, scale=10).items():
            v = [p.get(b) for b in s2_names]
            if all(x is not None for x in v):
                S2[pid, j] = v
        print(f'year {y}: done in {time.time() - t:.0f}s', flush=True)

    blocks = (np.floor(df.lon / GRID_DEG).astype(int).astype(str) + '_'
              + np.floor(df.lat / GRID_DEG).astype(int).astype(str))
    rng = np.random.default_rng(SEED)
    ub = blocks.unique()
    test_blocks = set(rng.choice(ub, size=int(0.3 * len(ub)), replace=False))
    split = np.where(blocks.isin(test_blocks), 'test', 'train')

    np.savez(OUT, AE=AE, S2=S2, s2_names=np.array(s2_names), years=np.array(YEARS),
             lon=df.lon.values, lat=df.lat.values, T=df['T'].values, strat=df.strat.values,
             block=blocks.values, split=split)
    print('saved', OUT, AE.shape, S2.shape)


if __name__ == '__main__':
    main()
