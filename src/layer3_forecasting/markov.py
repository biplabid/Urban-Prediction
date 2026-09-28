"""Markov Chain transition probability matrices from LULC time series."""

import ee
import numpy as np


def compute_transition_matrix(lulc_from, lulc_to, aoi, n_classes=4, scale=30):
    transition = lulc_from.multiply(10).add(lulc_to).rename('transition')
    pixel_area = ee.Image.pixelArea()

    matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            code = i * 10 + j
            area = transition.eq(code).multiply(pixel_area).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=aoi, scale=scale, maxPixels=1e9
            ).get('transition')
            matrix[i][j] = ee.Number(area).getInfo()

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums


def average_transition_matrix(matrices):
    return np.mean(list(matrices.values()), axis=0)


def project_class_areas(current_areas, tm, steps=1):
    areas = np.array(current_areas)
    for _ in range(steps):
        areas = areas @ tm
    return areas
