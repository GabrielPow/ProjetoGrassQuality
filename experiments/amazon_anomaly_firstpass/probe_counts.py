from common import *
init()
for T in EVENT_YEARS:
    h = strata_image(T).reduceRegion(ee.Reducer.frequencyHistogram(), aoi(), scale=300,
                                     maxPixels=1e10, tileScale=4).getInfo()
    print(T, h)
