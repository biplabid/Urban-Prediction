"""Cellular Automata forecast loop — iterative LULC prediction."""

import ee


def forecast_step(current_lulc, features, gbt_prob, markov_tm, aoi, n_classes=4, scale=30):
    suitability = features.classify(gbt_prob).rename('urban_prob')
    non_urban = current_lulc.neq(0)
    suitability = suitability.updateMask(non_urban)

    pixel_area = ee.Image.pixelArea()
    class_areas = {}
    for c in range(n_classes):
        a = current_lulc.eq(c).multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=scale, maxPixels=1e9
        ).get('lulc')
        class_areas[c] = ee.Number(a).getInfo()

    new_urban_m2 = sum(class_areas[c] * markov_tm[c][0] for c in range(1, n_classes))
    non_urban_total = sum(class_areas[c] for c in range(1, n_classes))
    new_urban_frac = new_urban_m2 / non_urban_total if non_urban_total > 0 else 0

    target_pctl = max(0, min(99, (1 - new_urban_frac) * 100))
    threshold = suitability.reduceRegion(
        reducer=ee.Reducer.percentile([target_pctl]),
        geometry=aoi, scale=scale, maxPixels=1e9
    ).get('urban_prob')
    threshold_val = ee.Number(threshold).getInfo()

    new_urban_mask = suitability.gt(threshold_val).unmask(0)
    water_protected = current_lulc.eq(2)
    new_urban_mask = new_urban_mask.And(water_protected.Not())

    next_lulc = current_lulc.where(non_urban.And(new_urban_mask), 0)
    return next_lulc, new_urban_m2
