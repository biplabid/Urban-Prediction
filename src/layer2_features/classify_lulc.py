"""LULC classification using GEE CART classifier with spectral-threshold pseudo-labels."""

import ee


LULC_CLASSES = {0: 'Urban', 1: 'Vegetation', 2: 'Water', 3: 'Barren'}
LULC_PALETTE = ['#e41a1c', '#4daf4a', '#377eb8', '#c2a06a']
CLASSIFY_BANDS = ['NDVI', 'NDBI', 'NDWI']


def generate_training_samples(composite, aoi, num_points=500, seed=42):
    ndvi = composite.select('NDVI')
    ndbi = composite.select('NDBI')
    ndwi = composite.select('NDWI')

    urban_mask = ndbi.gt(0.0).And(ndvi.lt(0.25))
    veg_mask = ndvi.gt(0.35).And(ndbi.lt(0.0))
    water_mask = ndwi.gt(0.1).And(ndvi.lt(0.15))
    barren_mask = ndvi.lt(0.15).And(ndbi.lt(0.05)).And(ndwi.lt(0.0))

    label_image = (
        ee.Image(0)
        .where(urban_mask, 0)
        .where(veg_mask, 1)
        .where(water_mask, 2)
        .where(barren_mask, 3)
        .rename('label')
        .toUint8()
    )
    any_class = urban_mask.Or(veg_mask).Or(water_mask).Or(barren_mask)
    label_image = label_image.updateMask(any_class)

    training_input = composite.addBands(label_image)
    return training_input.stratifiedSample(
        numPoints=num_points,
        classBand='label',
        region=aoi,
        scale=30,
        seed=seed,
        geometries=True,
    )


def train_classifier(training_samples, max_nodes=50):
    return ee.Classifier.smileCart(maxNodes=max_nodes).train(
        features=training_samples,
        classProperty='label',
        inputProperties=CLASSIFY_BANDS,
    )


def classify_composite(composite, classifier):
    return composite.select(CLASSIFY_BANDS).classify(classifier).toUint8().rename('lulc')
