"""
Layer 1 — Demographic proxy ingestion via Google Earth Engine.

Pulls two population density datasets for the Guwahati AOI:
  1. WorldPop — UN-adjusted population count at ~100m resolution (2000-2020)
  2. GHS-POP (GHSL) — Global Human Settlement population grid at ~100m (1975-2030)

These serve as urbanization-intensity covariates until the 2027 Census
provides a ground-truth refresh.

Usage:
    python -m src.layer1_ingestion.ingest_demographics --source worldpop --year 2020
"""
import argparse
import ee

from src.common.aoi import get_aoi_ee
from config.settings import GCS_BUCKET, GCS_RAW_PREFIX

WORLDPOP_COLLECTION = "WorldPop/GP/100m/pop_age_sex_cons_unadj"
GHSL_POP_COLLECTION = "JRC/GHSL/P2023A/GHS_POP"

WORLDPOP_YEARS = list(range(2000, 2021))
GHSL_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]


def build_worldpop(year: int) -> ee.Image:
    aoi = get_aoi_ee()
    image = (
        ee.ImageCollection(WORLDPOP_COLLECTION)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("year", year))
        .select("population")
        .mosaic()
        .clip(aoi)
    )
    return image


def build_ghsl_pop(epoch: int) -> ee.Image:
    aoi = get_aoi_ee()
    image = (
        ee.ImageCollection(GHSL_POP_COLLECTION)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("system:index", str(epoch)))
        .first()
        .select("population_count")
        .clip(aoi)
    )
    return image


def export_to_drive(
    image: ee.Image,
    name: str,
    scale: int = 100,
) -> ee.batch.Task:
    aoi = get_aoi_ee()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder="guwahati_urban_prediction",
        fileNamePrefix=name,
        region=aoi,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def export_to_gcs(
    image: ee.Image,
    name: str,
    bucket: str = GCS_BUCKET,
    scale: int = 100,
) -> ee.batch.Task:
    aoi = get_aoi_ee()
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=name,
        bucket=bucket,
        fileNamePrefix=f"{GCS_RAW_PREFIX}/demographics/{name}",
        region=aoi,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def ingest_demographics(
    source: str = "worldpop",
    year: int = 2020,
    destination: str = "drive",
    bucket: str = GCS_BUCKET,
) -> ee.batch.Task:
    if source == "worldpop":
        print(f"[Demographics] Building WorldPop population grid for {year}")
        image = build_worldpop(year)
        name = f"worldpop_{year}"
    elif source == "ghsl":
        print(f"[Demographics] Building GHS-POP grid for {year}")
        image = build_ghsl_pop(year)
        name = f"ghsl_pop_{year}"
    else:
        raise ValueError(f"Unknown source: {source}")

    if destination == "gcs":
        task = export_to_gcs(image, name, bucket=bucket)
    else:
        task = export_to_drive(image, name)

    print(f"[Demographics] Export task started: {task.status()['description']}")
    return task


def main():
    parser = argparse.ArgumentParser(
        description="Ingest demographic proxy data for Guwahati")
    parser.add_argument("--source", default="worldpop",
                        choices=["worldpop", "ghsl"])
    parser.add_argument("--year", required=True, type=int,
                        help="Year/epoch to pull")
    parser.add_argument("--dest", default="drive", choices=["gcs", "drive"])
    parser.add_argument("--bucket", default=GCS_BUCKET)
    args = parser.parse_args()

    ee.Initialize()
    ingest_demographics(
        source=args.source,
        year=args.year,
        destination=args.dest,
        bucket=args.bucket,
    )


if __name__ == "__main__":
    main()
