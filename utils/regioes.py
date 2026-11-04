"""Região alvo do projeto: a Amazônia brasileira.

O Earth Engine não tem um contorno público do bioma Amazônia (o do IBGE/MapBiomas não está
aberto). Aproximamos o bioma pelas ecorregiões RESOLVE (2017) de floresta amazônica. A parte
"brasileira" vem do próprio MapBiomas, que só tem dado dentro do Brasil. A diferença para o
bioma oficial fica nas bordas.
"""
import ee

ECOREGIOES_ASSET = 'RESOLVE/ECOREGIONS/2017'

# Ecorregiões de floresta tropical úmida (BIOME_NUM 1) da bacia amazônica e do escudo das
# Guianas com área no Brasil, por ECO_ID. Ficaram de fora as que também são BIOME_NUM 1 mas não
# são Amazônia (Mata Atlântica, restingas, enclaves da Caatinga, Fernando de Noronha) e as
# ecorregiões não florestais da borda (lavrado de Roraima, mangues da costa norte).
ECOREGIOES_AMAZONIA = {
    446: 'Caqueta moist forests',
    464: 'Guianan Highlands moist forests',
    465: 'Guianan lowland moist forests',
    466: 'Guianan piedmont moist forests',
    467: 'Gurupa várzea',
    469: 'Iquitos várzea',
    473: 'Japurá-Solimões-Negro moist forests',
    474: 'Juruá-Purus moist forests',
    476: 'Madeira-Tapajós moist forests',
    480: 'Marajó várzea',
    481: 'Mato Grosso tropical dry forests',
    482: 'Monte Alegre várzea',
    484: 'Negro-Branco moist forests',
    490: 'Pantepui forests & shrublands',
    496: 'Purus várzea',
    497: 'Purus-Madeira moist forests',
    498: 'Rio Negro campinarana',
    503: 'Solimões-Japurá moist forests',
    505: 'Southwest Amazon moist forests',
    507: 'Tapajós-Xingu moist forests',
    508: 'Tocantins/Pindare moist forests',
    511: 'Uatumã-Trombetas moist forests',
    518: 'Xingu-Tocantins-Araguaia moist forests',
}

# Retângulo (lon_min, lat_min, lon_max, lat_max) que contém a Amazônia brasileira inteira.
BBOX_AMAZONIA = (-74.0, -18.5, -43.5, 5.5)


def ecorregioes_amazonia():
    """FeatureCollection com as ecorregiões amazônicas (inclui a parte fora do Brasil)."""
    return (ee.FeatureCollection(ECOREGIOES_ASSET)
            .filter(ee.Filter.inList('ECO_ID', list(ECOREGIOES_AMAZONIA))))


def retangulo_amazonia():
    return ee.Geometry.Rectangle(list(BBOX_AMAZONIA))


def com_ecorregiao(pontos):
    """Mantém só os pontos dentro de uma ecorregião amazônica e anota o ECO_ID em 'eco_id'.

    `pontos` é uma FeatureCollection de pontos. Usa um join vetorial: é rápido para alguns
    milhares de pontos, ao contrário de rasterizar os polígonos das ecorregiões.
    """
    join = ee.Join.saveFirst('eco').apply(
        pontos, ecorregioes_amazonia(), ee.Filter.intersects(leftField='.geo', rightField='.geo'))
    return join.map(lambda f: f.set('eco_id', ee.Feature(f.get('eco')).get('ECO_ID')).set('eco', None))
