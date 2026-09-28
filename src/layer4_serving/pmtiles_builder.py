"""PMTiles generation from GeoJSON using tippecanoe."""

import os
import shutil
import subprocess


def build_pmtiles(geojson_files, output_path='osm_guwahati.pmtiles'):
    if not shutil.which('tippecanoe'):
        raise RuntimeError('tippecanoe not found. Install: apt-get install tippecanoe')

    layer_args = []
    for name, path in geojson_files.items():
        if os.path.exists(path):
            layer_args.extend(['-L', f'{name}:{path}'])

    if not layer_args:
        raise ValueError('No valid GeoJSON files provided')

    cmd = [
        'tippecanoe', '-o', output_path,
        '-zg', '--drop-densest-as-needed',
        '--extend-zooms-if-still-dropping',
        '--force',
    ] + layer_args

    subprocess.run(cmd, check=True)
    return output_path
