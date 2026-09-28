"""
Layer 1 — Sentinel-2 Surface Reflectance ingestion via Google Earth Engine.

Pulls cloud-masked, median-composited Sentinel-2 imagery for each configured
epoch over the Guwahati AOI.  Computes spectral indices (NDVI, NDBI, NDWI)
and exports multi-band GeoTIFFs to Cloud Storage (or Drive as fallback).

Usage (standalone):
    python -m src.layer1_ingestion.ingest_sentinel2 --epoch 2023

Usage (from orchestrator):
    from src.layer1_ingestion.ingest_sentinel2 import ingest_sentinel2
    ingest_sentinel2(epoch_label="2023")
"""
import argparse
import ee

from src.common.aoi import get_aoi_ee
from config.settings import (
    SENTINEL2_COLLECTION,
    CLOUD_COVER_MAX,
    PIXEL_SCALE_M,
    GCS_BUCKET,
    GCS_RAW_PREFIX,
    EPOCHS,
)


def _cloud_mask_s2(image: ee.Image) -> ee.Image:
    """Mask clouds and cirrus using the SCL band (Scene Classification Layer)."""
    scl = image.select("SCL")
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(clear)


def _add_indices(image: ee.Image) -> ee.Image:
    """Add NDVI, NDBI, and NDWI bands."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return image.addBands([ndvi, ndbi, ndwi])


def build_composite(start_date: str, end_date: str) -> ee.Image:
    """Build a cloud-free median composite with spectral indices."""
    aoi = get_aoi_ee()

    collection = (
        ee.ImageCollection(SENTINEL2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_MAX))
        .map(_cloud_mask_s2)
        .map(_add_indices)
    )

    bands = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "NDWI"]
    composite = collection.select(bands).median().clip(aoi)
    return composite


def export_to_gcs(
    image: ee.Image,
    epoch_label: str,
    bucket: str = GCS_BUCKET,
    scale: int = 10,
) -> ee.batch.Task:
    """Start an Earth Engine export task to Cloud Storage."""
    aoi = get_aoi_ee()
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f"sentinel2_{epoch_label}",
        bucket=bucket,
        fileNamePrefix=f"{GCS_RAW_PREFIX}/sentinel2/{epoch_label}",
        region=aoi,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def export_to_drive(
    image: ee.Image,
    epoch_label: str,
    scale: int = 10,
) -> ee.batch.Task:
    """Fallback: export to Google Drive if no GCS bucket is configured."""
    aoi = get_aoi_ee()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"sentinel2_{epoch_label}",
        folder="guwahati_urban_prediction",
        fileNamePrefix=f"sentinel2_{epoch_label}",
        region=aoi,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def ingest_sentinel2(
    epoch_label: str,
    destination: str = "drive",
    bucket: str = GCS_BUCKET,
) -> ee.batch.Task:
    """
    Full ingestion pipeline for one epoch.

    Args:
        epoch_label: Key into EPOCHS["historical"], e.g. "2023"
        destination: "gcs" or "drive"
        bucket: GCS bucket name (only used when destination="gcs")

    Returns:
        The started EE export task.
    """
    epoch = next(e for e in EPOCHS["historical"] if e["label"] == epoch_label)

    print(f"[Sentinel-2] Building composite for {epoch_label} "
          f"({epoch['start']} to {epoch['end']})")

    composite = build_composite(epoch["start"], epoch["end"])

    if destination == "gcs":
        task = export_to_gcs(composite, epoch_label, bucket=bucket)
    else:
        task = export_to_drive(composite, epoch_label)

    print(f"[Sentinel-2] Export task started: {task.status()['description']}")
    return task


def main():
    parser = argparse.ArgumentParser(description="Ingest Sentinel-2 imagery for Guwahati")
    parser.add_argument("--epoch", required=True, help="Epoch label, e.g. 2016, 2019, 2023")
    parser.add_argument("--dest", default="drive", choices=["gcs", "drive"],
                        help="Export destination (default: drive)")
    parser.add_argument("--bucket", default=GCS_BUCKET, help="GCS bucket name")
    args = parser.parse_args()

    ee.Initialize()
    ingest_sentinel2(args.epoch, destination=args.dest, bucket=args.bucket)


if __name__ == "__main__":
    main()
