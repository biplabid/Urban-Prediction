"""
Guwahati Area of Interest — shared geometry used by all ingestion scripts.

The AOI covers the Guwahati Metropolitan Area roughly bounded by:
  - West: Azara / LGB International Airport
  - East: Narengi / Noonmati refinery area
  - North: Brahmaputra riverbank
  - South: Jorabat / NH-37 corridor toward Meghalaya

All coordinates are WGS 84 (EPSG:4326).
"""
import ee

GUWAHATI_BBOX = {
    "west": 91.55,
    "south": 26.10,
    "east": 91.88,
    "north": 26.25,
}


def get_aoi_ee() -> ee.Geometry:
    b = GUWAHATI_BBOX
    return ee.Geometry.Rectangle([b["west"], b["south"], b["east"], b["north"]])


def get_aoi_bbox_tuple() -> tuple[float, float, float, float]:
    b = GUWAHATI_BBOX
    return (b["south"], b["west"], b["north"], b["east"])


def get_aoi_overpass() -> str:
    b = GUWAHATI_BBOX
    return f"{b['south']},{b['west']},{b['north']},{b['east']}"
