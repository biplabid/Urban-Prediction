"""
Project-wide configuration for Guwahati Urban Prediction Platform.
All paths, constants, and tunables live here.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TILES_DIR = DATA_DIR / "tiles"

GCS_BUCKET = "guwahati-urban-prediction"
GCS_RAW_PREFIX = "raw"

GUWAHATI_BBOX = {
    "west": 91.55,
    "south": 26.10,
    "east": 91.88,
    "north": 26.25,
}

GUWAHATI_CENTER = {"lon": 91.7362, "lat": 26.1445}

CRS = "EPSG:4326"
PIXEL_SCALE_M = 30

SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
LANDSAT8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
LANDSAT5_COLLECTION = "LANDSAT/LT05/C02/T1_L2"
VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"

LULC_CLASSES = {
    0: "urban",
    1: "vegetation",
    2: "water",
    3: "barren",
}

LULC_PALETTE = ["#e41a1c", "#4daf4a", "#377eb8", "#c2a06a"]

EPOCHS = {
    "historical": [
        {"label": "2016", "start": "2016-01-01", "end": "2016-12-31"},
        {"label": "2019", "start": "2019-01-01", "end": "2019-12-31"},
        {"label": "2023", "start": "2023-01-01", "end": "2023-12-31"},
    ],
    "backtest_train": {"label": "2016", "start": "2016-01-01", "end": "2016-12-31"},
    "backtest_val": {"label": "2021", "start": "2021-01-01", "end": "2021-12-31"},
}

CLOUD_COVER_MAX = 5

OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
