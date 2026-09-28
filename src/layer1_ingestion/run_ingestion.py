"""
Layer 1 — Orchestrator: runs the full ingestion pipeline for Guwahati.

Kicks off all data pulls in the right order:
  1. Sentinel-2 composites (3 historical epochs)
  2. Landsat composites (5 epochs for longer baseline)
  3. VIIRS night-lights (annual, 2014-2024)
  4. OSM roads and infrastructure (one-shot)
  5. Demographic proxies (WorldPop + GHSL)

GEE exports are async — this script starts them all and prints a status
summary.  Use `check_tasks()` to poll for completion.

Usage:
    python -m src.layer1_ingestion.run_ingestion --dest drive
    python -m src.layer1_ingestion.run_ingestion --only sentinel2 osm
"""
import argparse
import time

import ee

from src.layer1_ingestion.ingest_sentinel2 import ingest_sentinel2
from src.layer1_ingestion.ingest_landsat import ingest_landsat, LANDSAT_EPOCHS
from src.layer1_ingestion.ingest_viirs import ingest_viirs
from src.layer1_ingestion.ingest_osm import ingest_osm
from src.layer1_ingestion.ingest_demographics import ingest_demographics
from config.settings import EPOCHS, GCS_BUCKET


def run_all(
    destination: str = "drive",
    bucket: str = GCS_BUCKET,
    only: list[str] | None = None,
):
    """
    Run all Layer 1 ingestion pipelines.

    Args:
        destination: "gcs" or "drive" for GEE exports
        bucket: GCS bucket name
        only: Optional list of pipelines to run. Valid values:
              sentinel2, landsat, viirs, osm, demographics
    """
    pipelines = only or ["sentinel2", "landsat", "viirs", "osm", "demographics"]
    tasks = []

    if "sentinel2" in pipelines:
        print("\n" + "=" * 60)
        print("SENTINEL-2 INGESTION")
        print("=" * 60)
        for epoch in EPOCHS["historical"]:
            task = ingest_sentinel2(
                epoch["label"], destination=destination, bucket=bucket)
            tasks.append(("sentinel2", epoch["label"], task))

    if "landsat" in pipelines:
        print("\n" + "=" * 60)
        print("LANDSAT INGESTION")
        print("=" * 60)
        for epoch in LANDSAT_EPOCHS:
            task = ingest_landsat(
                epoch["label"], destination=destination, bucket=bucket)
            tasks.append(("landsat", epoch["label"], task))

    if "viirs" in pipelines:
        print("\n" + "=" * 60)
        print("VIIRS NIGHT-LIGHTS INGESTION")
        print("=" * 60)
        for year in [2016, 2019, 2021, 2023]:
            task = ingest_viirs(year, destination=destination, bucket=bucket)
            tasks.append(("viirs", str(year), task))

    if "osm" in pipelines:
        print("\n" + "=" * 60)
        print("OSM ROADS & INFRASTRUCTURE INGESTION")
        print("=" * 60)
        ingest_osm()

    if "demographics" in pipelines:
        print("\n" + "=" * 60)
        print("DEMOGRAPHIC PROXIES INGESTION")
        print("=" * 60)
        for year in [2000, 2010, 2015, 2020]:
            task = ingest_demographics(
                source="worldpop", year=year,
                destination=destination, bucket=bucket)
            tasks.append(("worldpop", str(year), task))

        for epoch in [2000, 2010, 2020]:
            task = ingest_demographics(
                source="ghsl", year=epoch,
                destination=destination, bucket=bucket)
            tasks.append(("ghsl", str(epoch), task))

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(tasks)} GEE export tasks started")
    print("=" * 60)
    for pipeline, label, task in tasks:
        status = task.status()
        print(f"  {pipeline}/{label}: {status['state']} "
              f"(id: {status.get('id', 'pending')})")

    if tasks:
        print("\nGEE exports are asynchronous. Run check_tasks() or visit")
        print("https://code.earthengine.google.com/tasks to monitor progress.")

    return tasks


def check_tasks(tasks: list | None = None):
    """Poll and print the status of all active EE tasks."""
    if tasks:
        for pipeline, label, task in tasks:
            status = task.status()
            print(f"  {pipeline}/{label}: {status['state']}")
    else:
        active = ee.batch.Task.list()
        for t in active[:20]:
            print(f"  {t.status()['description']}: {t.status()['state']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full Layer 1 ingestion for Guwahati")
    parser.add_argument("--dest", default="drive", choices=["gcs", "drive"],
                        help="Export destination for GEE tasks (default: drive)")
    parser.add_argument("--bucket", default=GCS_BUCKET)
    parser.add_argument("--only", nargs="*",
                        choices=["sentinel2", "landsat", "viirs", "osm", "demographics"],
                        help="Run only specific pipelines")
    args = parser.parse_args()

    ee.Initialize()
    run_all(destination=args.dest, bucket=args.bucket, only=args.only)


if __name__ == "__main__":
    main()
