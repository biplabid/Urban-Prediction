"""
Layer 1 — VIIRS DNB (Day/Night Band) monthly night-lights ingestion via GEE.

Night-light radiance is a proxy for economic activity and urbanization
intensity.  This script pulls annual median composites of the
avg_rad band for each epoch, clipped to the Guwahati AOI.

The output is a single-band GeoTIFF (radiance in nW/cm²/sr) at ~500m
native resolution, exported at 500m scale.

Usage:
    python -m src.layer1_ingestion.ingest_viirs --year 2023
"""
import argparse
import ee

from src.common.aoi import get_aoi_ee
from config.settings import (
    VIIRS_COLLECTION,
    GCS_BUCKET,
    GCS_RAW_PREFIX,
)

VIIRS_SCALE_M = 500

VIIRS_YEARS = list(range(2014, 2025))


def build_annual_composite(year: int) -> ee.Image:
    aoi = get_aoi_ee()

    collection = (
        ee.ImageCollection(VIIRS_COLLECTION)
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("avg_rad")
    )

    composite = collection.median().clip(aoi)
    return composite


def export_to_drive(image: ee.Image, year: int) -> ee.batch.Task:
    aoi = get_aoi_ee()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"viirs_{year}",
        folder="guwahati_urban_prediction",
        fileNamePrefix=f"viirs_{year}",
        region=aoi,
        scale=VIIRS_SCALE_M,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def export_to_gcs(
    image: ee.Image,
    year: int,
    bucket: str = GCS_BUCKET,
) -> ee.batch.Task:
    aoi = get_aoi_ee()
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f"viirs_{year}",
        bucket=bucket,
        fileNamePrefix=f"{GCS_RAW_PREFIX}/viirs/viirs_{year}",
        region=aoi,
        scale=VIIRS_SCALE_M,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def ingest_viirs(
    year: int,
    destination: str = "drive",
    bucket: str = GCS_BUCKET,
) -> ee.batch.Task:
    print(f"[VIIRS] Building night-light composite for {year}")

    composite = build_annual_composite(year)

    if destination == "gcs":
        task = export_to_gcs(composite, year, bucket=bucket)
    else:
        task = export_to_drive(composite, year)

    print(f"[VIIRS] Export task started: {task.status()['description']}")
    return task


def main():
    parser = argparse.ArgumentParser(description="Ingest VIIRS night-lights for Guwahati")
    parser.add_argument("--year", required=True, type=int,
                        help="Year to pull, e.g. 2023")
    parser.add_argument("--dest", default="drive", choices=["gcs", "drive"])
    parser.add_argument("--bucket", default=GCS_BUCKET)
    args = parser.parse_args()

    ee.Initialize()
    ingest_viirs(args.year, destination=args.dest, bucket=args.bucket)


if __name__ == "__main__":
    main()
