"""Gradient Boosted Trees model for urban growth suitability."""

import ee


FEATURE_BANDS = ['NDVI', 'NDBI', 'NDWI', 'dist_water', 'dist_urban_edge', 'dist_road', 'slope', 'elevation']


def build_change_samples(lulc_before, lulc_after, features, aoi, scale=30, n_pos=1000, n_neg=2000):
    became_urban = lulc_before.neq(0).And(lulc_after.eq(0)).rename('became_urban').toUint8()
    stayed_nonurban = lulc_before.neq(0).And(lulc_after.neq(0)).rename('became_urban')
    stayed_nonurban = stayed_nonurban.multiply(0).toUint8()

    pos_input = features.addBands(became_urban.updateMask(became_urban))
    neg_input = features.addBands(stayed_nonurban.updateMask(lulc_before.neq(0).And(lulc_after.neq(0))))

    pos = pos_input.sample(region=aoi, scale=scale, numPixels=n_pos, seed=42, geometries=True)
    neg = neg_input.sample(region=aoi, scale=scale, numPixels=n_neg, seed=43, geometries=True)
    return pos.merge(neg)


def train_gbt(samples, n_trees=100, shrinkage=0.1, max_nodes=20):
    gbt = ee.Classifier.smileGradientTreeBoost(
        numberOfTrees=n_trees, shrinkage=shrinkage, maxNodes=max_nodes, seed=42
    ).train(features=samples, classProperty='became_urban', inputProperties=FEATURE_BANDS)
    return gbt, gbt.setOutputMode('PROBABILITY')
