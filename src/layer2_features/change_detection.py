"""Post-classification change detection between LULC epochs."""

import ee


def compute_change(lulc_from, lulc_to):
    change = lulc_from.multiply(10).add(lulc_to).rename('change').toUint8()
    urban_gain = lulc_from.neq(0).And(lulc_to.eq(0)).rename('urban_gain').toUint8()
    urban_loss = lulc_from.eq(0).And(lulc_to.neq(0)).rename('urban_loss').toUint8()
    return change.addBands(urban_gain).addBands(urban_loss)


def urban_area_km2(lulc_image, aoi):
    pixel_area = ee.Image.pixelArea()
    urban_area = lulc_image.eq(0).multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=30,
        maxPixels=1e9,
    ).get('lulc')
    return ee.Number(urban_area).divide(1e6)
