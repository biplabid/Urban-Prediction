"""Proximity and terrain features for urban prediction."""

import ee


def distance_to_water(lulc_image, scale=30):
    water = lulc_image.eq(2)
    return water.fastDistanceTransform(256).sqrt().multiply(scale).rename('dist_water_m')


def distance_to_urban_edge(lulc_image, scale=30):
    urban = lulc_image.eq(0)
    edge = urban.And(urban.focal_min(1).Not())
    return edge.fastDistanceTransform(256).sqrt().multiply(scale).rename('dist_urban_edge_m')


def distance_to_roads(aoi, scale=30):
    viirs = (
        ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
        .filterBounds(aoi)
        .filterDate('2021-01-01', '2021-12-31')
        .select('avg_rad')
        .median()
        .clip(aoi)
    )
    lit_mask = viirs.gt(5).unmask(0)
    return lit_mask.fastDistanceTransform(256).sqrt().multiply(scale).rename('dist_road_m')


def terrain_features(aoi):
    srtm = ee.Image('USGS/SRTMGL1_003').clip(aoi)
    slope = ee.Terrain.slope(srtm).rename('slope_deg')
    elevation = srtm.select('elevation')
    return slope.addBands(elevation)


def build_proximity_stack(lulc_image, aoi, scale=30):
    return (
        distance_to_water(lulc_image, scale)
        .addBands(distance_to_urban_edge(lulc_image, scale))
        .addBands(distance_to_roads(aoi, scale))
        .addBands(terrain_features(aoi))
        .toFloat()
    )
