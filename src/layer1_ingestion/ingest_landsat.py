"""
Layer 1 — Landsat 5/8 Surface Reflectance ingestion via Google Earth Engine.

Landsat provides the longer historical baseline (back to the 1990s) that
Sentinel-2 cannot cover.  Used for the backtest epochs and early change
detection.  Produces 30m composites with the same spectral indices.

Usage:
    python -m src.layer1_ingestion.ingest_landsat --epoch 2011 --sensor landsat8
"""
import argparse
import ee

from src.common.aoi import get_aoi_ee
from config.settings import (
    LANDSAT8_COLLECTION,
    LANDSAT5_COLLECTION,
    CLOUD_COVER_MAX,
    PIXEL_SCALE_M,
    GCS_BUCKET,
    GCS_RAW_PREFIX,
)

LANDSAT_EPOCHS = [
    {"label": "2000", "start": "2000-01-01", "end": "2000-12-31", "sensor": "landsat5"},
    {"label": "2005", "start": "2005-01-01", "end": "2005-12-31", "sensor": "landsat5"},
    {"label": "2011", "start": "2011-01-01", "end": "2011-12-31", "sensor": "landsat5"},
    {"label": "2016", "start": "2016-01-01", "end": "2016-12-31", "sensor": "landsat8"},
    {"label": "2021", "start": "2021-01-01", "end": "2021-12-31", "sensor": "landsat8"},
]


def _cloud_mask_l8(image: ee.Image) -> ee.Image:
    """Mask clouds using QA_PIXEL band for Landsat 8 Collection 2."""
    qa = image.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud.And(shadow))


def _cloud_mask_l5(image: ee.Image) -> ee.Image:
    """Mask clouds using QA_PIXEL band for Landsat 5 Collection 2."""
    qa = image.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud.And(shadow))


def _scale_l8(image: ee.Image) -> ee.Image:
    """Apply Collection 2 Level 2 scaling factors for Landsat 8."""
    optical = image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]) \
        .multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True)


def _scale_l5(image: ee.Image) -> ee.Image:
    """Apply Collection 2 Level 2 scaling factors for Landsat 5."""
    optical = image.select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]) \
        .multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True)


def _add_indices_l8(image: ee.Image) -> ee.Image:
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndbi = image.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
    ndwi = image.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    return image.addBands([ndvi, ndbi, ndwi])


def _add_indices_l5(image: ee.Image) -> ee.Image:
    ndvi = image.normalizedDifference(["SR_B4", "SR_B3"]).rename("NDVI")
    ndbi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDBI")
    ndwi = image.normalizedDifference(["SR_B2", "SR_B4"]).rename("NDWI")
    return image.addBands([ndvi, ndbi, ndwi])


def build_composite(
    start_date: str,
    end_date: str,
    sensor: str = "landsat8",
) -> ee.Image:
    aoi = get_aoi_ee()

    if sensor == "landsat8":
        collection_id = LANDSAT8_COLLECTION
        mask_fn = _cloud_mask_l8
        scale_fn = _scale_l8
        index_fn = _add_indices_l8
        out_bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                     "NDVI", "NDBI", "NDWI"]
    else:
        collection_id = LANDSAT5_COLLECTION
        mask_fn = _cloud_mask_l5
        scale_fn = _scale_l5
        index_fn = _add_indices_l5
        out_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7",
                     "NDVI", "NDBI", "NDWI"]

    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_COVER_MAX))
        .map(mask_fn)
        .map(scale_fn)
        .map(index_fn)
    )

    composite = collection.select(out_bands).median().clip(aoi)
    return composite


def export_to_drive(
    image: ee.Image,
    epoch_label: str,
    sensor: str,
) -> ee.batch.Task:
    aoi = get_aoi_ee()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"{sensor}_{epoch_label}",
        folder="guwahati_urban_prediction",
        fileNamePrefix=f"{sensor}_{epoch_label}",
        region=aoi,
        scale=PIXEL_SCALE_M,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def export_to_gcs(
    image: ee.Image,
    epoch_label: str,
    sensor: str,
    bucket: str = GCS_BUCKET,
) -> ee.batch.Task:
    aoi = get_aoi_ee()
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f"{sensor}_{epoch_label}",
        bucket=bucket,
        fileNamePrefix=f"{GCS_RAW_PREFIX}/landsat/{sensor}_{epoch_label}",
        region=aoi,
        scale=PIXEL_SCALE_M,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def ingest_landsat(
    epoch_label: str,
    sensor: str | None = None,
    destination: str = "drive",
    bucket: str = GCS_BUCKET,
) -> ee.batch.Task:
    epoch = next(e for e in LANDSAT_EPOCHS if e["label"] == epoch_label)
    sensor = sensor or epoch["sensor"]

    print(f"[Landsat] Building {sensor} composite for {epoch_label} "
          f"({epoch['start']} to {epoch['end']})")

    composite = build_composite(epoch["start"], epoch["end"], sensor=sensor)

    if destination == "gcs":
        task = export_to_gcs(composite, epoch_label, sensor, bucket=bucket)
    else:
        task = export_to_drive(composite, epoch_label, sensor)

    print(f"[Landsat] Export task started: {task.status()['description']}")
    return task


def main():
    parser = argparse.ArgumentParser(description="Ingest Landsat imagery for Guwahati")
    parser.add_argument("--epoch", required=True, help="Epoch label, e.g. 2000, 2011, 2021")
    parser.add_argument("--sensor", choices=["landsat5", "landsat8"],
                        help="Override sensor selection")
    parser.add_argument("--dest", default="drive", choices=["gcs", "drive"])
    parser.add_argument("--bucket", default=GCS_BUCKET)
    args = parser.parse_args()

    ee.Initialize()
    ingest_landsat(args.epoch, sensor=args.sensor, destination=args.dest, bucket=args.bucket)


if __name__ == "__main__":
    main()
