"""
Layer 1 — Data Ingestion Notebook (Colab-compatible .py script)
================================================================

Run this in Google Colab or locally after authenticating with Earth Engine.

Steps:
  1. Authenticate and initialize Earth Engine
  2. Pull Sentinel-2 composites for 3 historical epochs
  3. Pull Landsat composites for 5 epochs (longer baseline)
  4. Pull VIIRS night-lights for key years
  5. Pull OSM roads and infrastructure (direct download)
  6. Pull WorldPop / GHSL demographic grids

All GEE exports go to Google Drive (free, no GCS bucket needed).
OSM data downloads directly as GeoJSON files.
"""

# %% [markdown]
# ## 0. Setup

# %%
# In Colab, install dependencies first:
# !pip install earthengine-api requests

import ee

# Authenticate — this opens a browser tab for OAuth
ee.Authenticate()
ee.Initialize(project="ee-guwahati-urban")  # replace with your GEE project ID

print("Earth Engine initialized:", ee.String("connected").getInfo())

# %% [markdown]
# ## 1. Verify AOI — Guwahati Metropolitan Area

# %%
BBOX = {"west": 91.55, "south": 26.10, "east": 91.88, "north": 26.25}
aoi = ee.Geometry.Rectangle([BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]])

area_km2 = aoi.area().divide(1e6).getInfo()
print(f"AOI area: {area_km2:.1f} km²")

# %% [markdown]
# ## 2. Sentinel-2 — 3 historical epochs at 10m

# %%
def cloud_mask_s2(image):
    scl = image.select("SCL")
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(clear)

def add_indices_s2(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return image.addBands([ndvi, ndbi, ndwi])

s2_epochs = [
    ("2016", "2016-01-01", "2016-12-31"),
    ("2019", "2019-01-01", "2019-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
]

s2_tasks = []
for label, start, end in s2_epochs:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
        .map(cloud_mask_s2)
        .map(add_indices_s2)
    )

    bands = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "NDWI"]
    composite = col.select(bands).median().clip(aoi)

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=f"sentinel2_{label}",
        folder="guwahati_urban_prediction",
        fileNamePrefix=f"sentinel2_{label}",
        region=aoi,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    s2_tasks.append(task)
    print(f"[S2] Export started: sentinel2_{label}")

print(f"\n{len(s2_tasks)} Sentinel-2 tasks submitted.")

# %% [markdown]
# ## 3. Landsat 5/8 — 5 epochs for longer baseline

# %%
def cloud_mask_l8(image):
    qa = image.select("QA_PIXEL")
    return image.updateMask(qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0)))

def scale_l8(image):
    optical = image.select(["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"]).multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True)

def cloud_mask_l5(image):
    qa = image.select("QA_PIXEL")
    return image.updateMask(qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0)))

def scale_l5(image):
    optical = image.select(["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"]).multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True)

landsat_epochs = [
    ("2000", "2000-01-01", "2000-12-31", "LANDSAT/LT05/C02/T1_L2", cloud_mask_l5, scale_l5,
     ["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"], ("SR_B4","SR_B3"), ("SR_B5","SR_B4"), ("SR_B2","SR_B4")),
    ("2005", "2005-01-01", "2005-12-31", "LANDSAT/LT05/C02/T1_L2", cloud_mask_l5, scale_l5,
     ["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"], ("SR_B4","SR_B3"), ("SR_B5","SR_B4"), ("SR_B2","SR_B4")),
    ("2011", "2011-01-01", "2011-12-31", "LANDSAT/LT05/C02/T1_L2", cloud_mask_l5, scale_l5,
     ["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"], ("SR_B4","SR_B3"), ("SR_B5","SR_B4"), ("SR_B2","SR_B4")),
    ("2016", "2016-01-01", "2016-12-31", "LANDSAT/LC08/C02/T1_L2", cloud_mask_l8, scale_l8,
     ["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"], ("SR_B5","SR_B4"), ("SR_B6","SR_B5"), ("SR_B3","SR_B5")),
    ("2021", "2021-01-01", "2021-12-31", "LANDSAT/LC08/C02/T1_L2", cloud_mask_l8, scale_l8,
     ["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"], ("SR_B5","SR_B4"), ("SR_B6","SR_B5"), ("SR_B3","SR_B5")),
]

ls_tasks = []
for label, start, end, collection, mask_fn, scale_fn, bands, nir_red, swir_nir, green_nir in landsat_epochs:
    col = (
        ee.ImageCollection(collection)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", 5))
        .map(mask_fn)
        .map(scale_fn)
    )

    def add_idx(image, nir_red=nir_red, swir_nir=swir_nir, green_nir=green_nir):
        ndvi = image.normalizedDifference(list(nir_red)).rename("NDVI")
        ndbi = image.normalizedDifference(list(swir_nir)).rename("NDBI")
        ndwi = image.normalizedDifference(list(green_nir)).rename("NDWI")
        return image.addBands([ndvi, ndbi, ndwi])

    col = col.map(add_idx)
    out_bands = bands + ["NDVI", "NDBI", "NDWI"]
    composite = col.select(out_bands).median().clip(aoi)

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=f"landsat_{label}",
        folder="guwahati_urban_prediction",
        fileNamePrefix=f"landsat_{label}",
        region=aoi,
        scale=30,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    ls_tasks.append(task)
    print(f"[LS] Export started: landsat_{label}")

print(f"\n{len(ls_tasks)} Landsat tasks submitted.")

# %% [markdown]
# ## 4. VIIRS Night-Lights — annual composites

# %%
viirs_tasks = []
for year in [2016, 2019, 2021, 2023]:
    col = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("avg_rad")
    )
    composite = col.median().clip(aoi)

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=f"viirs_{year}",
        folder="guwahati_urban_prediction",
        fileNamePrefix=f"viirs_{year}",
        region=aoi,
        scale=500,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    viirs_tasks.append(task)
    print(f"[VIIRS] Export started: viirs_{year}")

print(f"\n{len(viirs_tasks)} VIIRS tasks submitted.")

# %% [markdown]
# ## 5. OSM Roads & Infrastructure (direct download)

# %%
import requests
import json
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
bbox_str = f"{BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']}"

osm_queries = {
    "roads_major": f"""[out:json][timeout:120];
        (way["highway"~"^(motorway|trunk|primary|secondary)$"]({bbox_str}););
        out geom;""",
    "roads_all": f"""[out:json][timeout:120];
        (way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified)$"]({bbox_str}););
        out geom;""",
    "railways": f"""[out:json][timeout:120];
        (way["railway"="rail"]({bbox_str}););
        out geom;""",
    "infrastructure": f"""[out:json][timeout:120];
        (node["amenity"~"^(hospital|school|college|university|marketplace)$"]({bbox_str});
         node["office"="government"]({bbox_str}););
        out center;""",
    "water_bodies": f"""[out:json][timeout:120];
        (way["waterway"~"^(river|stream|canal)$"]({bbox_str});
         way["natural"="water"]({bbox_str}););
        out geom;""",
}

import time

for name, query in osm_queries.items():
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    features = []
    for el in elements:
        props = el.get("tags", {})
        props["osm_id"] = el.get("id")
        if el["type"] == "node":
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif "geometry" in el:
            coords = [[p["lon"], p["lat"]] for p in el["geometry"]]
            geom = {"type": "LineString", "coordinates": coords}
        elif "center" in el:
            geom = {"type": "Point", "coordinates": [el["center"]["lon"], el["center"]["lat"]]}
        else:
            continue
        features.append({"type": "Feature", "geometry": geom, "properties": props})

    geojson = {"type": "FeatureCollection", "features": features}
    out_path = f"osm_{name}.geojson"
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"[OSM] {name}: {len(features)} features → {out_path}")
    time.sleep(2)

# %% [markdown]
# ## 6. Demographics — WorldPop population density

# %%
demo_tasks = []
for year in [2000, 2010, 2015, 2020]:
    image = (
        ee.ImageCollection("WorldPop/GP/100m/pop_age_sex_cons_unadj")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("year", year))
        .select("population")
        .mosaic()
        .clip(aoi)
    )

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"worldpop_{year}",
        folder="guwahati_urban_prediction",
        fileNamePrefix=f"worldpop_{year}",
        region=aoi,
        scale=100,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    demo_tasks.append(task)
    print(f"[Demo] Export started: worldpop_{year}")

print(f"\n{len(demo_tasks)} demographic tasks submitted.")

# %% [markdown]
# ## 7. Check all task status

# %%
import time

all_tasks = s2_tasks + ls_tasks + viirs_tasks + demo_tasks
print(f"Total GEE export tasks: {len(all_tasks)}")
print("Monitor at: https://code.earthengine.google.com/tasks\n")

for t in all_tasks:
    status = t.status()
    print(f"  {status['description']}: {status['state']}")
