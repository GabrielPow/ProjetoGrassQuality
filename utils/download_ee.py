"""Download de recortes Sentinel-2 + MapBiomas da Amazônia brasileira via Earth Engine.

Uso no notebook:

    from utils.download_ee import baixar_amostra_amazonia, carregar_amostra
    baixar_amostra_amazonia(nome='amazonia_exploracao')
    X, mapbiomas, anos, info = carregar_amostra('amazonia_exploracao')

O download salva o dado cru: a imagem e o MapBiomas completo (todas as classes, todos os anos),
sem categoria nem filtro. O critério de sorteio fica gravado só como origem da amostra (onde
olhamos), não como rótulo (o que a área é).

Critérios de sorteio, avaliados no pixel de 10 m do MapBiomas onde cai o ponto:
- 'uniforme': qualquer ponto da Amazônia brasileira, com chance proporcional à área.
- 'mudou': o pixel teve mais de uma classe do MapBiomas entre os anos pedidos.
- 'mineracao': o pixel é mineração (classe 30) em pelo menos um dos anos.

'uniforme' e 'mudou' saem do mesmo sorteio de pontos ao acaso. Mineração é rara demais para
isso (menos de 0,1% da área), então primeiro buscamos candidatos num MapBiomas agregado em
blocos de `escala_busca_mineracao` metros (classe mais comum do bloco) e depois confirmamos o
critério no pixel de 10 m. Consequência: garimpos muito pequenos têm menos chance de sair.
"""
import json
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from .regioes import BBOX_AMAZONIA, ECOREGIOES_AMAZONIA, com_ecorregiao, retangulo_amazonia

EE_PROJECT_ID = 'projetograssquality'
RAIZ_DADOS = Path(__file__).resolve().parent.parent / 'data' / 'raw'

S2_ASSET = 'COPERNICUS/S2_SR_HARMONIZED'
CLOUD_SCORE_ASSET = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
BANDAS_S2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
# As 12 bandas do Sentinel-2 L2A na ordem em que o TerraMind foi pré-treinado (B10 não existe no L2A).
# As de 20 m e 60 m são reamostradas para a grade de 10 m (vizinho mais próximo).
BANDAS_S2_TODAS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
# MapBiomas 10 m (coleção 4, feito com Sentinel-2): bandas classification_2017 ... classification_2025
MAPBIOMAS_ASSET = ('projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/'
                   'mapbiomas_10m_collection4_coverage_v1')
CLASSE_MINERACAO = 30
ESCALA_M = 10
CRITERIOS = ('uniforme', 'mineracao', 'mudou')


def inicializar_ee():
    # Só Initialize: o login vem da credencial do gcloud (ADC). ee.Authenticate() não é
    # necessário e, no Windows, abre um login que o Google bloqueia.
    ee.Initialize(project=EE_PROJECT_ID)


def mapbiomas(anos):
    """MapBiomas 10 m com uma banda por ano, chamadas c2019, c2020, ..."""
    return ee.Image(MAPBIOMAS_ASSET).select([f'classification_{a}' for a in anos],
                                            [f'c{a}' for a in anos])


def composto_s2(area, ano, limiar_nuvem=0.6, bandas=BANDAS_S2):
    """Composto anual do Sentinel-2: mediana, pixel a pixel, de todas as passagens do ano depois
    de apagar as nuvens (Cloud Score+ abaixo de `limiar_nuvem`).

    Mesma receita do s2_image() do experimento do Gabriel
    (experiments/amazon_anomaly_firstpass/common.py), mas na escala original do Sentinel-2
    (refletância × 10000, uint16), que ocupa metade do espaço. Pixels sem nenhuma observação
    limpa no ano viram 0.
    """
    s2 = (ee.ImageCollection(S2_ASSET).filterBounds(area)
          .filterDate(f'{ano}-01-01', f'{ano + 1}-01-01')
          .linkCollection(ee.ImageCollection(CLOUD_SCORE_ASSET), ['cs_cdf'])
          .map(lambda im: im.updateMask(im.select('cs_cdf').gte(limiar_nuvem))))
    return s2.select(list(bandas)).median().unmask(0).round().toUint16()


# ----------------------------------------------------------------------------- sorteio

def _tentar(func, tentativas=4):
    """Chama func() repetindo em caso de erro transitório do Earth Engine (cota, timeout)."""
    for k in range(tentativas):
        try:
            return func()
        except Exception as e:
            if k == tentativas - 1:
                raise
            espera = 5 * 2 ** k
            print(f'    erro do Earth Engine ({str(e)[:80]}), tentando de novo em {espera}s')
            time.sleep(espera)


def _pontos_uniformes(n, rng):
    """n pontos ao acaso no retângulo da Amazônia, com chance proporcional à área.

    Longitude uniforme e seno da latitude uniforme: assim cada km² tem a mesma chance
    (sortear a latitude direto daria peso maior às latitudes altas).
    """
    lon_min, lat_min, lon_max, lat_max = BBOX_AMAZONIA
    lon = rng.uniform(lon_min, lon_max, n)
    seno = rng.uniform(np.sin(np.radians(lat_min)), np.sin(np.radians(lat_max)), n)
    return lon, np.degrees(np.arcsin(seno))


def _avaliar_pontos(lon, lat, anos):
    """Ecorregião e classe do MapBiomas (10 m) em cada ano, para cada ponto.

    Devolve só os pontos dentro da Amazônia brasileira (ecorregião amazônica e com dado do
    MapBiomas, que só existe no Brasil), na mesma ordem da entrada.
    """
    colunas = [f'c{a}' for a in anos]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'i': i})
                               for i, (x, y) in enumerate(zip(lon, lat))])
    amostra = mapbiomas(anos).sampleRegions(collection=com_ecorregiao(fc), properties=['i', 'eco_id'],
                                            scale=ESCALA_M, tileScale=4)
    linhas = []
    for f in _tentar(amostra.getInfo)['features']:
        p = f['properties']
        if all(c in p for c in colunas):
            i = int(p['i'])
            linhas.append({'lon': lon[i], 'lat': lat[i], 'eco_id': int(p['eco_id']),
                           **{c: int(p[c]) for c in colunas}, 'i': i})
    df = pd.DataFrame(linhas, columns=['lon', 'lat', 'eco_id', *colunas, 'i'])
    return df.sort_values('i').drop(columns='i').reset_index(drop=True)


def _candidatos_mineracao(n, anos, seed, escala_busca):
    """Busca grossa de pontos onde o MapBiomas marca mineração em algum ano."""
    tem_mineracao = (mapbiomas(anos).eq(CLASSE_MINERACAO).reduce(ee.Reducer.max())
                     .selfMask().rename('m').toInt())
    fc = tem_mineracao.stratifiedSample(numPoints=n, classBand='m', region=retangulo_amazonia(),
                                        scale=escala_busca, seed=seed, geometries=True, tileScale=16)
    coords = np.array([f['geometry']['coordinates'] for f in _tentar(fc.getInfo)['features']])
    return coords[:, 0], coords[:, 1]


def _distancia_km(lon1, lat1, lon2, lat2):
    """Distância em linha reta sobre a Terra (fórmula de haversine), em km."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _adicionar(escolhidos, candidatos, n, criterio, dist_min_km):
    """Percorre os candidatos (já em ordem aleatória) e guarda os n primeiros que ficam a pelo
    menos dist_min_km de todos os pontos já escolhidos, para não baixar o mesmo lugar duas vezes."""
    novos = 0
    for _, c in candidatos.iterrows():
        if novos == n:
            break
        if all(_distancia_km(c.lon, c.lat, e['lon'], e['lat']) >= dist_min_km for e in escolhidos):
            escolhidos.append({'criterio': criterio, **c.to_dict()})
            novos += 1
    if novos < n:
        print(f'Aviso: só {novos} de {n} pontos encontrados para o critério {criterio!r}.')


def sortear_pontos(n_por_criterio, anos, seed=42, dist_min_km=10, escala_busca_mineracao=300,
                   tamanho_lote=1500, max_lotes=10):
    """Sorteia os pontos de cada critério. Mesmo seed e parâmetros = mesmos pontos.

    Devolve uma tabela com critério, lon, lat, ecorregião e a classe do MapBiomas no ponto em
    cada ano (c2019, c2020, ...).
    """
    desconhecidos = set(n_por_criterio) - set(CRITERIOS)
    if desconhecidos:
        raise ValueError(f'Critérios desconhecidos: {sorted(desconhecidos)}. Use {CRITERIOS}.')
    rng = np.random.default_rng(seed)
    colunas = [f'c{a}' for a in anos]
    n = {c: n_por_criterio.get(c, 0) for c in CRITERIOS}

    candidatos = {}
    if n['uniforme'] or n['mudou']:
        # Lotes de pontos ao acaso até ter folga de 5x candidatos (a distância mínima descarta alguns)
        lotes = []
        for _ in range(max_lotes):
            lotes.append(_avaliar_pontos(*_pontos_uniformes(tamanho_lote, rng), anos))
            pool = pd.concat(lotes, ignore_index=True)
            mudou = pool[colunas].nunique(axis=1) > 1
            if len(pool) >= 5 * n['uniforme'] and mudou.sum() >= 5 * n['mudou']:
                break
        candidatos['uniforme'] = pool
        candidatos['mudou'] = pool[mudou]
    if n['mineracao']:
        lon, lat = _candidatos_mineracao(max(100, 30 * n['mineracao']), anos, seed, escala_busca_mineracao)
        cand = _avaliar_pontos(lon, lat, anos)
        cand = cand[(cand[colunas] == CLASSE_MINERACAO).any(axis=1)]   # confirma no pixel de 10 m
        candidatos['mineracao'] = cand.iloc[rng.permutation(len(cand))]

    escolhidos = []
    for criterio, k in n_por_criterio.items():
        if k:
            _adicionar(escolhidos, candidatos[criterio], k, criterio, dist_min_km)
    pontos = pd.DataFrame(escolhidos)
    pontos[['eco_id', *colunas]] = pontos[['eco_id', *colunas]].astype(int)
    pontos['ecorregiao'] = pontos['eco_id'].map(ECOREGIOES_AMAZONIA)
    return pontos[['criterio', 'lon', 'lat', 'eco_id', 'ecorregiao', *colunas]]


# ----------------------------------------------------------------------------- download

def _grade_utm(lon, lat, tamanho_px):
    """Grade de 10 m em UTM com o ponto dentro do pixel [tamanho_px // 2, tamanho_px // 2].

    UTM tem pixels quadrados de 10 m de verdade, e os cantos em múltiplos de 10 m coincidem com
    a grade do Sentinel-2, então as bandas de 10 m não são reamostradas.
    """
    zona = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zona
    x, y = _tentar(ee.Geometry.Point([lon, lat]).transform(f'EPSG:{epsg}', 0.01).coordinates().getInfo)
    meio = tamanho_px // 2 * ESCALA_M
    x0 = np.floor(x / ESCALA_M) * ESCALA_M - meio
    y0 = np.ceil(y / ESCALA_M) * ESCALA_M + meio
    return {'dimensions': {'width': tamanho_px, 'height': tamanho_px},
            'affineTransform': {'scaleX': ESCALA_M, 'shearX': 0, 'translateX': float(x0),
                                'shearY': 0, 'scaleY': -ESCALA_M, 'translateY': float(y0)},
            'crsCode': f'EPSG:{epsg}'}


def _baixar_pixels(imagem, grade):
    """Baixa a imagem na grade pedida, direto como numpy (H, W, bandas)."""
    a = _tentar(lambda: ee.data.computePixels(
        {'expression': imagem, 'fileFormat': 'NUMPY_NDARRAY', 'grid': grade}))
    return np.stack([a[b] for b in a.dtype.names], axis=-1)


def baixar_recorte(lon, lat, anos, tamanho_px=256, limiar_nuvem=0.6, bandas=BANDAS_S2):
    """Recorte centrado em (lon, lat): Sentinel-2 (anos, H, W, bandas) e MapBiomas (anos, H, W)."""
    grade = _grade_utm(lon, lat, tamanho_px)
    t, lado = grade['affineTransform'], tamanho_px * ESCALA_M
    area = ee.Geometry.Rectangle([t['translateX'], t['translateY'] - lado,
                                  t['translateX'] + lado, t['translateY']], grade['crsCode'], False)
    s2 = np.stack([_baixar_pixels(composto_s2(area, a, limiar_nuvem, bandas), grade) for a in anos])
    mb = _baixar_pixels(mapbiomas(anos).unmask(0).toUint8(), grade)
    return s2, np.moveaxis(mb, -1, 0), grade


def baixar_amostra_amazonia(nome='amazonia_exploracao', n_por_criterio=None, anos=range(2019, 2026),
                            tamanho_px=256, seed=42, limiar_nuvem=0.6, dist_min_km=10,
                            escala_busca_mineracao=300, bandas=BANDAS_S2):
    """Sorteia pontos na Amazônia brasileira e salva um recorte por ponto em data/raw/<nome>/.

    n_por_criterio: quantos recortes de cada critério (ver o topo do arquivo). Padrão:
        {'uniforme': 6, 'mineracao': 3, 'mudou': 3}.
    bandas: bandas do Sentinel-2. BANDAS_S2_TODAS dá as 12 que o TerraMind espera. Os pontos
        sorteados não dependem das bandas: mesmo seed e critérios = mesmos lugares.

    Cada arquivo <k>_<criterio>.npz tem:
        s2         (anos, H, W, bandas) uint16  refletância × 10000; 0 = sem observação limpa no ano
        mapbiomas  (anos, H, W)    uint8   código de classe do MapBiomas 10 m; 0 = sem dado
        anos, bandas, lon, lat, ecorregiao, criterio, crs, transform (origem e georreferência)

    O ponto sorteado fica no pixel [H // 2, W // 2]. Rodar de novo com os mesmos parâmetros
    retoma um download interrompido e não baixa de novo o que já existe.
    Devolve uma tabela com os recortes.
    """
    if n_por_criterio is None:
        n_por_criterio = {'uniforme': 6, 'mineracao': 3, 'mudou': 3}
    anos, bandas = list(anos), list(bandas)
    parametros = {'n_por_criterio': n_por_criterio, 'anos': anos, 'tamanho_px': tamanho_px,
                  'seed': seed, 'limiar_nuvem': limiar_nuvem, 'dist_min_km': dist_min_km,
                  'escala_busca_mineracao': escala_busca_mineracao}
    if bandas != BANDAS_S2:   # só grava quando muda, para as amostras antigas continuarem válidas
        parametros['bandas'] = bandas
    parametros = json.dumps(parametros, sort_keys=True)
    pasta = RAIZ_DADOS / nome
    pasta.mkdir(parents=True, exist_ok=True)

    existentes = sorted(pasta.glob('*.npz'))
    for arq in existentes:
        with np.load(arq) as d:
            if str(d['parametros']) != parametros:
                raise ValueError(f'{pasta} já tem recortes de um sorteio com outros parâmetros. '
                                 'Use outro nome ou apague a pasta.')
    total = sum(n_por_criterio.values())
    if len(existentes) == total:
        print(f'Os {total} recortes já estão em {pasta}')
        return _ler_info(existentes)

    inicializar_ee()
    print('Sorteando pontos...')
    t = time.time()
    pontos = sortear_pontos(n_por_criterio, anos, seed, dist_min_km, escala_busca_mineracao)
    print(f'{len(pontos)} pontos sorteados ({time.time() - t:.0f}s). Baixando recortes...')

    for k, p in enumerate(pontos.itertuples()):
        arquivo = pasta / f'{k:02d}_{p.criterio}.npz'
        rotulo = f'[{k + 1:2d}/{len(pontos)}] {p.criterio:9s} ({p.lat:7.3f}, {p.lon:8.3f})  {p.ecorregiao}'
        if arquivo.exists():
            print(rotulo, '-> já baixado')
            continue
        t = time.time()
        s2, mb, grade = baixar_recorte(p.lon, p.lat, anos, tamanho_px, limiar_nuvem, bandas)
        af = grade['affineTransform']
        np.savez_compressed(arquivo, s2=s2, mapbiomas=mb, anos=np.array(anos), bandas=np.array(bandas),
                            lon=p.lon, lat=p.lat, ecorregiao=p.ecorregiao, criterio=p.criterio,
                            crs=grade['crsCode'], parametros=parametros,
                            transform=np.array([af['translateX'], ESCALA_M, 0, af['translateY'], 0, -ESCALA_M]))
        print(rotulo, f'-> ok ({time.time() - t:.0f}s)')
    return _ler_info(sorted(pasta.glob('*.npz')))


# ----------------------------------------------------------------------------- leitura

def _ler_info(arquivos):
    linhas = []
    for arq in arquivos:
        with np.load(arq) as d:
            linhas.append({'arquivo': arq.name, 'criterio': str(d['criterio']), 'lon': float(d['lon']),
                           'lat': float(d['lat']), 'ecorregiao': str(d['ecorregiao'])})
    return pd.DataFrame(linhas)


def carregar_amostra(nome):
    """Lê os recortes de data/raw/<nome>/.

    Devolve X (N, anos, H, W, 6) uint16, mapbiomas (N, anos, H, W) uint8, a lista de anos e uma
    tabela com a origem de cada recorte (arquivo, critério, lon, lat, ecorregião). Os nomes das
    bandas de X ficam em info.attrs['bandas'].
    """
    arquivos = sorted((RAIZ_DADOS / nome).glob('*.npz'))
    if not arquivos:
        raise FileNotFoundError(f'Nenhum recorte em {RAIZ_DADOS / nome}')
    X, mb = [], []
    for arq in arquivos:
        with np.load(arq) as d:
            X.append(d['s2'])
            mb.append(d['mapbiomas'])
            anos = d['anos'].tolist()
            bandas = d['bandas'].tolist()
    info = _ler_info(arquivos)
    info.attrs['bandas'] = bandas
    return np.stack(X), np.stack(mb), anos, info
