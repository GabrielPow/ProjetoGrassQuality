"""Shared config for the Amazon anomaly first-pass (AlphaEarth vs spectral change)."""
import ee

EE_PROJECT_ID = 'projetograssquality'
SEED = 42

# Tapajos / BR-163 (Para): deforestation front + artisanal gold mining (garimpo).
AOI_BOUNDS = [-57.0, -7.5, -55.0, -5.5]
EVENT_YEARS = [2021, 2022, 2023, 2024]   # T; history = T-2, T-1; confirmation = T+1 (<= 2025)
HIST = 2

LULC_ASSET = 'projects/mapbiomas-public/assets/brazil/lulc/v1'
AE_ASSET = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
AE_BANDS = [f'A{i:02d}' for i in range(64)]
S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

FOREST = [3]
PASTURE = [15]
MINING = [30]
# anthropic land use after clearing (pasture, mosaic, agriculture, urban, other non-veg, mining)
ANTHROPIC = [15, 21, 9, 18, 19, 20, 39, 40, 41, 62, 36, 46, 47, 48, 24, 25, 30]

STRATA = {0: 'stable_forest', 1: 'stable_pasture', 2: 'deforest', 3: 'mining'}
ANOMALY_STRATA = [2, 3]


def init():
    ee.Initialize(project=EE_PROJECT_ID)


def aoi():
    return ee.Geometry.Rectangle(AOI_BOUNDS)


def lulc(year):
    return (ee.ImageCollection(LULC_ASSET).filter(ee.Filter.eq('year', year))
            .first().select('classification'))


def isin(img, codes):
    return img.remap(codes, [1] * len(codes), 0)


def all_of(imgs):
    out = imgs[0]
    for im in imgs[1:]:
        out = out.And(im)
    return out


def strata_image(T):
    hist = list(range(T - HIST, T))
    after = [T, T + 1]
    stable_forest = all_of([isin(lulc(y), FOREST) for y in hist + after])
    stable_pasture = all_of([isin(lulc(y), PASTURE) for y in hist + after])
    was_forest = all_of([isin(lulc(y), FOREST) for y in hist])
    mining = was_forest.And(all_of([isin(lulc(y), MINING) for y in after]))
    deforest = was_forest.And(all_of([isin(lulc(y), ANTHROPIC) for y in after])).And(mining.Not())
    strat = (stable_forest.multiply(0).add(stable_pasture.multiply(1))
             .add(deforest.multiply(2)).add(mining.multiply(3)))
    valid = stable_forest.Or(stable_pasture).Or(deforest).Or(mining)
    return strat.rename('strat').updateMask(valid)


def ae_image(year):
    return (ee.ImageCollection(AE_ASSET).filterDate(f'{year}-01-01', f'{year + 1}-01-01')
            .mosaic().select(AE_BANDS))


def s2_image(year):
    """Annual cloud-masked (Cloud Score+) Sentinel-2 SR median + NDVI/NBR."""
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(aoi()).filterDate(f'{year}-01-01', f'{year + 1}-01-01'))
    cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2 = s2.linkCollection(cs, ['cs_cdf']).map(
        lambda im: im.updateMask(im.select('cs_cdf').gte(0.6)))
    med = s2.select(S2_BANDS).median().divide(10000)
    ndvi = med.normalizedDifference(['B8', 'B4']).rename('NDVI')
    nbr = med.normalizedDifference(['B8', 'B12']).rename('NBR')
    return med.addBands(ndvi).addBands(nbr)
