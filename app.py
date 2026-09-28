"""Cloud Run backend for Guwahati Urban Prediction Platform.

Serves the frontend and provides an API that generates fresh GEE tile URLs
on demand, solving the ~24h expiration problem.
"""

import json
import os

import ee
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="public", static_url_path="")

BBOX = {"west": 91.55, "south": 26.10, "east": 91.88, "north": 26.25}
SCALE = 30
LULC_PALETTE = ["e41a1c", "4daf4a", "377eb8", "c2a06a"]
CLASSIFY_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                  "NDVI", "NDBI", "NDWI"]

EPOCHS_CFG = {
    "2000": ("2000-11-01", "2001-04-30", "L5"),
    "2005": ("2005-11-01", "2006-04-30", "L5"),
    "2010": ("2010-11-01", "2011-04-30", "L5"),
    "2016": ("2016-11-01", "2017-04-30", "L8"),
    "2021": ("2021-11-01", "2022-04-30", "L8"),
}

_initialized = False


def init_ee():
    global _initialized
    if _initialized:
        return
    project = os.environ.get("GEE_PROJECT", "vivid-brand-182819")
    import google.auth
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/earthengine"]
    )
    ee.Initialize(credentials=credentials, project=project)
    _initialized = True


def cloud_mask_landsat(image):
    qa = image.select("QA_PIXEL")
    return image.updateMask(qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0)))


def add_indices(image, nir, red, swir, green):
    return image.addBands([
        image.normalizedDifference([nir, red]).rename("NDVI"),
        image.normalizedDifference([swir, nir]).rename("NDBI"),
        image.normalizedDifference([green, nir]).rename("NDWI"),
    ])


L8_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
L5_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
HARMONIZE_SLOPES  = [0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]
HARMONIZE_OFFSETS = [0.0095, 0.0064, 0.0044, -0.0119, -0.0026, -0.0015]


def scale_l8(image):
    return image.addBands(
        image.select(L8_BANDS).multiply(0.0000275).add(-0.2), overwrite=True)


def scale_and_harmonize_l5(image):
    scaled = image.select(L5_BANDS).multiply(0.0000275).add(-0.2)
    harmonized = (scaled
                  .multiply(HARMONIZE_SLOPES)
                  .add(HARMONIZE_OFFSETS)
                  .rename(L8_BANDS))
    return image.addBands(harmonized, overwrite=True)


def build_composite(aoi, start, end, sensor):
    if sensor == "L8":
        col_id = "LANDSAT/LC08/C02/T1_L2"
        sfn = scale_l8
    else:
        col_id = "LANDSAT/LT05/C02/T1_L2"
        sfn = scale_and_harmonize_l5

    nir, red, swir, green = "SR_B5", "SR_B4", "SR_B6", "SR_B3"

    def _add(img):
        return add_indices(img, nir, red, swir, green)

    col = (ee.ImageCollection(col_id)
           .filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUD_COVER", 30))
           .map(cloud_mask_landsat).map(sfn).map(_add))
    return col.select(L8_BANDS + ["NDVI", "NDBI", "NDWI"]).median().clip(aoi).toFloat()


DT_NEIGHBORHOOD = 128


def proximity_stack(aoi, lulc_img):
    dw = lulc_img.eq(2).fastDistanceTransform(DT_NEIGHBORHOOD).sqrt().multiply(SCALE).rename("dist_water")
    edge = lulc_img.eq(0).And(lulc_img.eq(0).focal_min(1).Not())
    de = edge.fastDistanceTransform(DT_NEIGHBORHOOD).sqrt().multiply(SCALE).rename("dist_urban_edge")
    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterBounds(aoi).filterDate("2021-01-01", "2021-12-31")
             .select("avg_rad").median().clip(aoi))
    dr = viirs.gt(5).unmask(0).fastDistanceTransform(DT_NEIGHBORHOOD).sqrt().multiply(SCALE).rename("dist_road")
    srtm = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    return (dw.addBands(de).addBands(dr)
            .addBands(ee.Terrain.slope(srtm).rename("slope"))
            .addBands(srtm.select("elevation")).toFloat())


def run_pipeline():
    """Run the full GEE pipeline and return tile URLs + stats."""
    import numpy as np

    aoi = ee.Geometry.Rectangle([BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]])

    composites = {}
    for lbl, (s, e, sen) in EPOCHS_CFG.items():
        composites[lbl] = build_composite(aoi, s, e, sen)

    def classify_epoch(comp):
        ndvi, ndbi, ndwi = comp.select("NDVI"), comp.select("NDBI"), comp.select("NDWI")
        is_water = ndwi.gt(0.1).And(ndvi.lt(0.15))
        is_veg = ndvi.gt(0.3).And(ndbi.lt(0))
        is_urban = ndbi.gt(0.05).And(ndvi.lt(0.2))
        is_barren = ndvi.lt(0.3).And(ndbi.lte(0.05)).And(ndbi.gt(-0.15)).And(ndwi.lt(0))

        labels = (ee.Image(0)
                  .where(is_urban, 0)
                  .where(is_veg, 1)
                  .where(is_barren, 3)
                  .where(is_water, 2)
                  .rename("label").toUint8())
        mask = is_urban.Or(is_veg).Or(is_water).Or(is_barren)
        samples = (comp.addBands(labels.updateMask(mask))
                   .stratifiedSample(500, "label", aoi, SCALE, seed=42, geometries=True))
        cart = ee.Classifier.smileCart(maxNodes=50).train(samples, "label", CLASSIFY_BANDS)
        return comp.select(CLASSIFY_BANDS).classify(cart).toUint8().rename("lulc")

    lulc_maps = {lbl: classify_epoch(comp) for lbl, comp in composites.items()}

    # Markov + GBT forecasting
    def compute_tm(lf, lt):
        tr = lf.multiply(10).add(lt).rename("t")
        pa = ee.Image.pixelArea()
        m = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                m[i][j] = ee.Number(
                    tr.eq(i * 10 + j).multiply(pa)
                    .reduceRegion(ee.Reducer.sum(), aoi, SCALE, maxPixels=1e9).get("t")
                ).getInfo()
        rs = m.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return m / rs

    tms = {}
    for e1, e2 in [("2000", "2005"), ("2005", "2010"), ("2010", "2016"), ("2016", "2021")]:
        tms[f"{e1}_{e2}"] = compute_tm(lulc_maps[e1], lulc_maps[e2])
    avg_tm = np.mean(list(tms.values()), axis=0)

    FEAT_BANDS = ["NDVI", "NDBI", "NDWI", "dist_water", "dist_urban_edge", "dist_road", "slope", "elevation"]

    def chg_samples(lf, la, comp, li):
        prox = proximity_stack(aoi, li)
        feat = comp.select(["NDVI", "NDBI", "NDWI"]).addBands(prox)
        bu = lf.neq(0).And(la.eq(0)).rename("became_urban").toUint8()
        sn = lf.neq(0).And(la.neq(0)).rename("became_urban").multiply(0).toUint8()
        return (feat.addBands(bu.updateMask(bu))
                .sample(aoi, SCALE, numPixels=1000, seed=42, geometries=True)
                .merge(feat.addBands(sn.updateMask(lf.neq(0).And(la.neq(0))))
                       .sample(aoi, SCALE, numPixels=2000, seed=43, geometries=True)))

    s1 = chg_samples(lulc_maps["2010"], lulc_maps["2016"], composites["2010"], lulc_maps["2010"])
    s2 = chg_samples(lulc_maps["2016"], lulc_maps["2021"], composites["2016"], lulc_maps["2016"])
    gbt = ee.Classifier.smileGradientTreeBoost(100, shrinkage=0.1, maxNodes=20, seed=42).train(
        s1.merge(s2), "became_urban", FEAT_BANDS)
    gbt_prob = gbt.setOutputMode("PROBABILITY")

    # Suitability is a static surface derived from the 2021 baseline. Computing
    # it once keeps each forecast step a cheap threshold on a fixed image rather
    # than nesting the previous step's whole computation graph.
    base_suit = (composites["2021"].select(["NDVI", "NDBI", "NDWI"])
                 .addBands(proximity_stack(aoi, lulc_maps["2021"]))
                 .classify(gbt_prob).rename("urban_prob").toFloat())

    suit_sample = base_suit.sample(region=aoi, scale=SCALE * 2, numPixels=8000,
                                   seed=1, dropNulls=True, tileScale=4)
    suit_vals = sorted(
        f["properties"]["urban_prob"]
        for f in suit_sample.getInfo()["features"]
        if f["properties"].get("urban_prob") is not None)

    def class_areas(img):
        hist = img.rename("lulc").reduceRegion(
            ee.Reducer.frequencyHistogram(), aoi, SCALE * 2,
            maxPixels=1e9, tileScale=4).get("lulc").getInfo() or {}
        px_area = (SCALE * 2) ** 2
        return {c: hist.get(str(c), 0) * px_area for c in range(4)}

    def forecast_step(cur, tm):
        ca = class_areas(cur)
        nu = sum(ca[c] * tm[c][0] for c in range(1, 4))
        nt = sum(ca[c] for c in range(1, 4))
        frac = nu / nt if nt > 0 else 0
        pctl = max(0.0, min(99.0, (1 - frac) * 100))
        if not suit_vals:
            return cur
        idx = min(len(suit_vals) - 1, int(pctl / 100 * len(suit_vals)))
        thr = suit_vals[idx]
        return cur.where(cur.neq(0).And(cur.neq(2)).And(base_suit.gt(thr).unmask(0)), 0)

    predicted = {}
    cur = lulc_maps["2021"]
    for yr in ["2026", "2031", "2036"]:
        cur = forecast_step(cur, avg_tm)
        predicted[yr] = cur

    all_lulc = {**lulc_maps, **predicted}
    lulc_vis = {"min": 0, "max": 3, "palette": LULC_PALETTE}

    tile_layers = {}

    for year in sorted(all_lulc.keys(), key=int):
        mi = all_lulc[year].getMapId(lulc_vis)
        tile_layers[f"lulc_{year}"] = {
            "url": mi["tile_fetcher"].url_format,
            "type": "raster", "group": "lulc",
            "label": f"LULC {year}" + (" (predicted)" if int(year) > 2021 else ""),
            "opacity": 0.7
        }

    for e1, e2 in [("2000", "2005"), ("2005", "2010"), ("2010", "2016"),
                   ("2016", "2021"), ("2021", "2026"), ("2026", "2031"), ("2031", "2036")]:
        prev, curr = all_lulc.get(e1), all_lulc.get(e2)
        if prev and curr:
            gain = prev.neq(0).And(curr.eq(0)).selfMask()
            color = "ff0000" if int(e2) <= 2021 else "ff6600"
            mi = gain.getMapId({"palette": [color]})
            tile_layers[f"urban_gain_{e1}_{e2}"] = {
                "url": mi["tile_fetcher"].url_format,
                "type": "raster", "group": "change",
                "label": f"Urban Growth {e1}→{e2}", "opacity": 0.8
            }

    viirs_vis = {"min": 0, "max": 60, "palette": ["000000", "0000ff", "00ffff", "ffff00", "ff0000"]}
    for y in [2016, 2019, 2021, 2023]:
        v = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterBounds(aoi).filterDate(f"{y}-01-01", f"{y}-12-31")
             .select("avg_rad").median().clip(aoi))
        tile_layers[f"viirs_{y}"] = {
            "url": v.getMapId(viirs_vis)["tile_fetcher"].url_format,
            "type": "raster", "group": "nightlights",
            "label": f"Night Lights {y}", "opacity": 0.7
        }

    tile_layers["suitability_2021"] = {
        "url": base_suit.getMapId({"min": 0, "max": 1, "palette": ["1a9850", "91cf60", "d9ef8b", "fee08b", "fc8d59", "d73027"]})["tile_fetcher"].url_format,
        "type": "raster", "group": "analysis", "label": "Urban Growth Suitability", "opacity": 0.7
    }
    tile_layers["ndvi_2021"] = {
        "url": composites["2021"].select("NDVI").getMapId({"min": -0.1, "max": 0.8, "palette": ["d73027", "fc8d59", "fee08b", "d9ef8b", "91cf60", "1a9850"]})["tile_fetcher"].url_format,
        "type": "raster", "group": "analysis", "label": "Vegetation (NDVI)", "opacity": 0.7
    }
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    tile_layers["elevation"] = {
        "url": dem.getMapId({"min": 40, "max": 300, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]})["tile_fetcher"].url_format,
        "type": "raster", "group": "terrain", "label": "Elevation (SRTM)", "opacity": 0.6
    }
    tile_layers["slope"] = {
        "url": ee.Terrain.slope(dem).getMapId({"min": 0, "max": 30, "palette": ["00ff00", "ffff00", "ff0000"]})["tile_fetcher"].url_format,
        "type": "raster", "group": "terrain", "label": "Slope", "opacity": 0.6
    }

    urban_stats = {}
    for year in sorted(all_lulc.keys(), key=int):
        a = ee.Number(
            all_lulc[year].eq(0).multiply(ee.Image.pixelArea())
            .reduceRegion(ee.Reducer.sum(), aoi, SCALE, maxPixels=1e9).get("lulc")
        ).divide(1e6).getInfo()
        urban_stats[year] = round(a, 2)

    return {
        "project": "Guwahati Urban Development Prediction Platform",
        "version": "1.0.0-poc",
        "center": [26.1445, 91.7362],
        "zoom": 12,
        "bounds": [BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]],
        "epochs": {
            "historical": ["2000", "2005", "2010", "2016", "2021"],
            "predicted": ["2026", "2031", "2036"],
        },
        "urban_area_km2": urban_stats,
        "lulc": {"classes": ["Urban", "Vegetation", "Water", "Barren"],
                 "palette": [f"#{c}" for c in LULC_PALETTE]},
        "tile_layers": tile_layers,
        "vector_layers": {},
    }


_cached_manifest = None


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/api/manifest")
def api_manifest():
    """Generate fresh GEE tile URLs. Results are cached in memory for the
    lifetime of this Cloud Run instance (~15 min idle timeout)."""
    global _cached_manifest
    try:
        init_ee()
        if _cached_manifest is None:
            _cached_manifest = run_pipeline()
        return jsonify(_cached_manifest)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        app.logger.error(f"Pipeline error: {tb}")
        return jsonify({"error": str(exc), "traceback": tb}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/ee-test")
def ee_test():
    """Quick test of EE authentication — no heavy pipeline."""
    try:
        init_ee()
        dem = ee.Image("USGS/SRTMGL1_003")
        val = dem.sample(ee.Geometry.Point(91.7362, 26.1445), 30).first().get("elevation").getInfo()
        return jsonify({"status": "ok", "elevation_at_guwahati": val})
    except Exception as exc:
        import traceback
        return jsonify({"status": "error", "error": str(exc), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
