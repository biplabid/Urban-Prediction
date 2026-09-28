"""
Layer 1 — OpenStreetMap roads and infrastructure ingestion via Overpass API.

Pulls road network, railway lines, and key infrastructure points (hospitals,
schools, markets) for the Guwahati AOI.  Saves as GeoJSON files that Layer 2
will rasterize into proximity features.

Usage:
    python -m src.layer1_ingestion.ingest_osm
"""
import json
import time
import argparse
from pathlib import Path

import requests

from src.common.aoi import get_aoi_overpass
from config.settings import OSM_OVERPASS_URL, RAW_DIR

OUTPUT_DIR = RAW_DIR / "osm"

QUERIES = {
    "roads_major": {
        "description": "Primary, secondary, trunk, and motorway roads",
        "query": """
            [out:json][timeout:120];
            (
              way["highway"~"^(motorway|trunk|primary|secondary)$"]({bbox});
            );
            out geom;
        """,
    },
    "roads_all": {
        "description": "All driveable roads including tertiary and residential",
        "query": """
            [out:json][timeout:120];
            (
              way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]({bbox});
            );
            out geom;
        """,
    },
    "railways": {
        "description": "Railway lines",
        "query": """
            [out:json][timeout:120];
            (
              way["railway"="rail"]({bbox});
            );
            out geom;
        """,
    },
    "infrastructure": {
        "description": "Key POI: hospitals, schools, markets, government offices",
        "query": """
            [out:json][timeout:120];
            (
              node["amenity"~"^(hospital|school|college|university|marketplace)$"]({bbox});
              node["office"="government"]({bbox});
              way["amenity"~"^(hospital|school|college|university|marketplace)$"]({bbox});
            );
            out center;
        """,
    },
    "water_bodies": {
        "description": "Rivers, streams, lakes, ponds — for flood-risk constraint",
        "query": """
            [out:json][timeout:120];
            (
              way["waterway"~"^(river|stream|canal)$"]({bbox});
              relation["waterway"="river"]({bbox});
              way["natural"="water"]({bbox});
              relation["natural"="water"]({bbox});
            );
            out geom;
        """,
    },
}


def _overpass_to_geojson(elements: list[dict]) -> dict:
    """Convert Overpass JSON elements to a GeoJSON FeatureCollection."""
    features = []
    for el in elements:
        props = {k: v for k, v in el.get("tags", {}).items()}
        props["osm_id"] = el.get("id")
        props["osm_type"] = el.get("type")

        if el["type"] == "node":
            geometry = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif el["type"] == "way" and "geometry" in el:
            coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
            geometry = {"type": "LineString", "coordinates": coords}
        elif el["type"] == "way" and "center" in el:
            geometry = {"type": "Point",
                        "coordinates": [el["center"]["lon"], el["center"]["lat"]]}
        elif el["type"] == "relation" and "center" in el:
            geometry = {"type": "Point",
                        "coordinates": [el["center"]["lon"], el["center"]["lat"]]}
        else:
            continue

        features.append({"type": "Feature", "geometry": geometry, "properties": props})

    return {"type": "FeatureCollection", "features": features}


def fetch_osm_layer(layer_name: str, bbox_str: str) -> dict:
    """Run one Overpass query and return a GeoJSON FeatureCollection."""
    spec = QUERIES[layer_name]
    query = spec["query"].replace("{bbox}", bbox_str)

    print(f"[OSM] Fetching {layer_name}: {spec['description']}")

    response = requests.post(
        OSM_OVERPASS_URL,
        data={"data": query},
        timeout=180,
    )
    response.raise_for_status()

    data = response.json()
    geojson = _overpass_to_geojson(data.get("elements", []))
    print(f"[OSM] {layer_name}: {len(geojson['features'])} features")
    return geojson


def ingest_osm(layers: list[str] | None = None) -> dict[str, Path]:
    """
    Fetch OSM layers for Guwahati and save as GeoJSON files.

    Returns:
        Dict mapping layer name to saved file path.
    """
    bbox_str = get_aoi_overpass()
    layers = layers or list(QUERIES.keys())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = {}
    for i, name in enumerate(layers):
        if i > 0:
            time.sleep(2)

        geojson = fetch_osm_layer(name, bbox_str)
        out_path = OUTPUT_DIR / f"{name}.geojson"
        out_path.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
        saved[name] = out_path
        print(f"[OSM] Saved {out_path}")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Ingest OSM data for Guwahati")
    parser.add_argument("--layers", nargs="*", choices=list(QUERIES.keys()),
                        help="Specific layers to fetch (default: all)")
    args = parser.parse_args()

    ingest_osm(layers=args.layers)


if __name__ == "__main__":
    main()
