"""GEE tile URL generation and manifest building."""

import ee
import json


LULC_PALETTE = ['e41a1c', '4daf4a', '377eb8', 'c2a06a']
LULC_LABELS = ['Urban', 'Vegetation', 'Water', 'Barren']


def get_lulc_tile_url(lulc_image):
    vis = {'min': 0, 'max': 3, 'palette': LULC_PALETTE}
    map_id = lulc_image.getMapId(vis)
    return map_id['tile_fetcher'].url_format


def get_suitability_tile_url(suit_image):
    vis = {'min': 0, 'max': 1,
           'palette': ['1a9850', '91cf60', 'd9ef8b', 'fee08b', 'fc8d59', 'd73027']}
    map_id = suit_image.getMapId(vis)
    return map_id['tile_fetcher'].url_format


def get_viirs_tile_url(viirs_image):
    vis = {'min': 0, 'max': 60,
           'palette': ['000000', '0000ff', '00ffff', 'ffff00', 'ff0000']}
    map_id = viirs_image.getMapId(vis)
    return map_id['tile_fetcher'].url_format


def build_manifest(tile_urls, urban_stats, vector_layers=None):
    return {
        'project': 'Guwahati Urban Development Prediction Platform',
        'version': '1.0.0-poc',
        'center': [26.1445, 91.7362],
        'zoom': 12,
        'bounds': [91.55, 26.10, 91.88, 26.25],
        'urban_area_km2': urban_stats,
        'lulc': {'classes': LULC_LABELS, 'palette': ['#' + c for c in LULC_PALETTE]},
        'tile_layers': tile_urls,
        'vector_layers': vector_layers or {},
    }


def save_manifest(manifest, path='tile_manifest.json'):
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
