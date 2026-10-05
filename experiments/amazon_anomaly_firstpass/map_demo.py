"""Wall-to-wall anomaly map for one window/year: AlphaEarth cosine change vs MapBiomas disturbances.

Computed server-side in Earth Engine; downloads small PNG thumbnails and composes one figure.
"""
import io
import sys
import urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from common import *

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
DIM = 768

init()
WINDOW = ee.Geometry.Rectangle([-55.75, -7.30, -55.35, -6.90])   # BR-163 corridor near Novo Progresso
hist = ae_image(T - 2).add(ae_image(T - 1))
hist = hist.divide(hist.pow(2).reduce('sum').sqrt())
cur = ae_image(T)
ae_score = ee.Image(1).subtract(cur.multiply(hist).reduce('sum')).rename('score')

s2h = s2_image(T - 2).select('NDVI').add(s2_image(T - 1).select('NDVI')).divide(2)
ndvi_drop = s2h.subtract(s2_image(T).select('NDVI'))

was_forest = isin(lulc(T - 2), FOREST).And(isin(lulc(T - 1), FOREST))
disturbed = was_forest.And(isin(lulc(T), FOREST).Not())

rgb = s2_image(T).select(['B4', 'B3', 'B2'])

panels = [
    (f'Sentinel-2 RGB {T}', rgb.visualize(min=0, max=0.15)),
    (f'MapBiomas: forest lost in {T}',
     ee.Image(1).visualize(palette=['f4f4f2']).blend(disturbed.selfMask().visualize(palette=['b3261e']))),
    (f'AlphaEarth cosine change {T - 2}-{T - 1} -> {T}', ae_score.visualize(min=0, max=0.6, palette=['0d0887', '7e03a8', 'cc4778', 'f89540', 'f0f921'])),
    (f'NDVI drop {T - 2}-{T - 1} -> {T}', ndvi_drop.visualize(min=0, max=0.4, palette=['0d0887', '7e03a8', 'cc4778', 'f89540', 'f0f921'])),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, (title, img) in zip(axes, panels):
    url = img.getThumbURL({'region': WINDOW, 'dimensions': DIM, 'format': 'png'})
    ax.imshow(Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=300).read())))
    ax.set_title(title, fontsize=9); ax.axis('off')
fig.suptitle('BR-163 near Novo Progresso (PA), ~44 x 44 km. Scale: AE 0-0.6, NDVI drop 0-0.4 (dark = low, yellow = high)', fontsize=9)
fig.tight_layout()
fig.savefig(f'results/map_{T}.png', dpi=130)
print('saved', f'results/map_{T}.png')
