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
INDEX_BANDS = ["NDVI", "NDBI", "NDWI", "MNDWI", "UI", "BSI", "SAVI"]
CLASSIFY_BANDS = INDEX_BANDS

# GHSL runs on a 5-year grid, so the 2016 and 2021 epochs map to its nearest
# steps. built_surface is m2 of built cover within each 100m cell.
GHSL_EPOCHS = {"2000": "2000", "2005": "2005", "2010": "2010",
               "2016": "2015", "2021": "2020"}
GHSL_CELL_AREA = 100 * 100
GHSL_BUILT_THRESHOLD = 0.2

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


def add_indices(image, nir, red, swir, green, blue, swir2):
    """Normalized ratios only. Raw reflectance shifts with atmosphere and
    illumination between epochs, so a classifier trained on one year's raw
    values does not transfer to another; ratios largely cancel that out.
    BSI is what separates bare soil from built-up, which NDBI alone cannot."""
    b = {"nir": image.select(nir), "red": image.select(red),
         "swir": image.select(swir), "green": image.select(green),
         "blue": image.select(blue), "swir2": image.select(swir2)}
    bsi = (b["swir"].add(b["red"]).subtract(b["nir"].add(b["blue"]))
           .divide(b["swir"].add(b["red"]).add(b["nir"]).add(b["blue"]))
           .rename("BSI"))
    savi = (b["nir"].subtract(b["red"]).multiply(1.5)
            .divide(b["nir"].add(b["red"]).add(0.5)).rename("SAVI"))
    return image.addBands([
        image.normalizedDifference([nir, red]).rename("NDVI"),
        image.normalizedDifference([swir, nir]).rename("NDBI"),
        image.normalizedDifference([green, nir]).rename("NDWI"),
        image.normalizedDifference([green, swir]).rename("MNDWI"),
        image.normalizedDifference([swir2, nir]).rename("UI"),
        bsi, savi,
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

    def _add(img):
        return add_indices(img, "SR_B5", "SR_B4", "SR_B6", "SR_B3", "SR_B2", "SR_B7")

    col = (ee.ImageCollection(col_id)
           .filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUD_COVER", 30))
           .map(cloud_mask_landsat).map(sfn).map(_add))
    return col.select(INDEX_BANDS).median().clip(aoi).toFloat()


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

    # Index thresholds cannot separate built-up from dry-season bare farmland:
    # both read high-NDBI/low-NDVI. Train on ESA WorldCover's validated Built-up
    # class instead, then apply that classifier to every harmonized epoch.
    wc = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(aoi)
    wc_label = (ee.Image(0)
                .where(wc.eq(50), 0)                                   # built-up
                .where(wc.eq(10).Or(wc.eq(20)).Or(wc.eq(30))
                       .Or(wc.eq(40)).Or(wc.eq(95)), 1)                # vegetation
                .where(wc.eq(80).Or(wc.eq(90)), 2)                     # water
                .where(wc.eq(60), 3)                                   # bare / sparse
                .rename("label").toUint8())
    wc_mask = (wc.eq(50).Or(wc.eq(10)).Or(wc.eq(20)).Or(wc.eq(30))
               .Or(wc.eq(40)).Or(wc.eq(95)).Or(wc.eq(80)).Or(wc.eq(90)).Or(wc.eq(60)))

    train_samples = (composites["2021"].addBands(wc_label.updateMask(wc_mask))
                     .stratifiedSample(600, "label", aoi, SCALE, seed=42,
                                       geometries=True, tileScale=4))
    cart = ee.Classifier.smileRandomForest(30, seed=42).train(
        train_samples, "label", CLASSIFY_BANDS)

    # The Landsat classifier supplies vegetation/water/barren context, but its
    # urban class was not comparable across epochs: self-classified built-up
    # broke at the L5/L8 boundary and drifted even within one sensor. Take the
    # urban class from GHSL instead, a multi-temporal built-up surface product
    # built for consistent change detection, and overlay it on the Landsat
    # classes. Non-urban labels are kept from Landsat at its finer 30m detail.
    ghsl_col = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
    available = set(ghsl_col.aggregate_array("system:index").getInfo() or [])
    missing = {v for v in GHSL_EPOCHS.values() if v not in available}
    if missing:
        raise RuntimeError(
            f"GHSL epochs {sorted(missing)} not in collection; available: {sorted(available)}")

    def built_fraction(epoch_year):
        img = ghsl_col.filter(
            ee.Filter.eq("system:index", GHSL_EPOCHS[epoch_year])).first()
        return (ee.Image(img).select("built_surface").clip(aoi)
                .divide(GHSL_CELL_AREA).clamp(0, 1).rename("built_frac"))

    lulc_maps = {}
    for lbl, comp in composites.items():
        landsat_cls = comp.select(CLASSIFY_BANDS).classify(cart).toUint8()
        is_built = built_fraction(lbl).gte(GHSL_BUILT_THRESHOLD)
        # Demote Landsat's own urban guess to barren, then apply GHSL built-up.
        base = landsat_cls.where(landsat_cls.eq(0), 3)
        lulc_maps[lbl] = (base.where(is_built, 0)
                          .where(landsat_cls.eq(2).And(is_built.Not()), 2)
                          .toUint8().rename("lulc"))

    # Markov + GBT forecasting. One frequencyHistogram per transition rather
    # than a reduceRegion per matrix cell: 16 round trips collapse to 1.
    def compute_tm(lf, lt):
        tr = lf.multiply(10).add(lt).rename("t")
        hist = tr.reduceRegion(
            ee.Reducer.frequencyHistogram(), aoi, SCALE * 2,
            maxPixels=1e9, tileScale=4).get("t").getInfo() or {}
        m = np.zeros((4, 4))
        for k, v in hist.items():
            code = int(float(k))
            i, j = divmod(code, 10)
            if 0 <= i < 4 and 0 <= j < 4:
                m[i][j] = float(v)
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

    # Cells eligible to urbanise: non-urban in 2021, with water protected.
    base_2021 = lulc_maps["2021"]
    candidates = base_2021.neq(0).And(base_2021.neq(2))

    # Histogram counts are pixels; the reported stats sum pixelArea(), which is
    # true ground area and runs ~12% under the nominal SCALE^2 at this latitude.
    # Calibrate one effective pixel area from the AOI so counts and reported
    # areas agree, instead of assuming 900 m2 per pixel.
    aoi_area = ee.Image.pixelArea().reduceRegion(
        ee.Reducer.sum(), aoi, SCALE, maxPixels=1e9, tileScale=8).get("area").getInfo()
    aoi_px = ee.Image(1).rename("n").reduceRegion(
        ee.Reducer.count(), aoi, SCALE, maxPixels=1e9, tileScale=8).get("n").getInfo()
    if not aoi_area or not aoi_px:
        raise RuntimeError("Could not calibrate pixel area over the AOI")
    PX_AREA = float(aoi_area) / float(aoi_px)

    hist_raw = base_suit.updateMask(candidates).reduceRegion(
        ee.Reducer.fixedHistogram(0, 1, 2000), aoi, SCALE,
        maxPixels=1e9, tileScale=8).get("urban_prob").getInfo()
    if not hist_raw:
        raise RuntimeError("Empty suitability histogram over candidate cells")
    # [[binLeft, count], ...] ascending; walk from the top to convert the most
    # suitable land first.
    suit_bins = [(float(b), float(c)) for b, c in hist_raw]
    BIN_W = 1.0 / len(suit_bins)

    def threshold_for_area(target_area):
        """Suitability cutoff enclosing target_area of candidate land.

        Interpolates inside the bin that straddles the target: returning the
        bin's left edge would admit every pixel in it, and near the peak of the
        distribution a single bin holds hundreds of km2.
        """
        acc = 0.0
        for left, count in reversed(suit_bins):
            area = count * PX_AREA
            if acc + area >= target_area and area > 0:
                take = (target_area - acc) / area
                return left + BIN_W * (1.0 - take)
            acc += area
        return suit_bins[0][0]

    def class_areas(img):
        hist = img.rename("lulc").reduceRegion(
            ee.Reducer.frequencyHistogram(), aoi, SCALE,
            maxPixels=1e9, tileScale=8).get("lulc").getInfo() or {}
        return {c: float(hist.get(str(c), 0)) * PX_AREA for c in range(4)}

    # Suitability is static, so each step must claim a progressively larger
    # share of the candidate pool. Thresholding on a per-step fraction would
    # reselect the same cells every time and stall the forecast.
    predicted = {}
    cur = base_2021
    areas_2021 = class_areas(base_2021)
    convertible = areas_2021[1] + areas_2021[3]
    cum_converted = 0.0

    step_targets = []
    for yr in ["2026", "2031", "2036"]:
        ca = class_areas(cur)
        cum_converted += sum(ca[c] * avg_tm[c][0] for c in (1, 3))
        step_targets.append(cum_converted)
        thr = threshold_for_area(min(cum_converted, convertible))
        cur = base_2021.where(candidates.And(base_suit.gte(thr)), 0)
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

    # All eight epochs in one reduction rather than one round trip each.
    years_sorted = sorted(all_lulc.keys(), key=int)
    urban_stack = ee.Image.cat([
        all_lulc[y].eq(0).multiply(ee.Image.pixelArea()).rename(f"u{y}")
        for y in years_sorted])
    areas = urban_stack.reduceRegion(
        ee.Reducer.sum(), aoi, SCALE, maxPixels=1e9, tileScale=4).getInfo() or {}
    urban_stats = {y: round((areas.get(f"u{y}") or 0) / 1e6, 2) for y in years_sorted}

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
        "diagnostics": {
            "urban_source": "JRC/GHSL/P2023A/GHS_BUILT_S",
            "ghsl_epoch_map": GHSL_EPOCHS,
            "ghsl_built_threshold": GHSL_BUILT_THRESHOLD,
            "transition_matrix_avg": [[round(v, 6) for v in row] for row in avg_tm.tolist()],
            "p_to_urban": {"vegetation": round(float(avg_tm[1][0]), 6),
                           "barren": round(float(avg_tm[3][0]), 6)},
            "areas_2021_km2": {c: round(areas_2021[c] / 1e6, 2) for c in areas_2021},
            "convertible_km2": round(convertible / 1e6, 2),
            "forecast_step_targets_km2": [round(t / 1e6, 2) for t in step_targets],
        },
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
