"""
tests/test_clipper.py
----------------------
Verifies that `src.data.clipper` correctly computes spatial intersections
and clips datasets to the intersection bounding box.

Run from the project root:
    python tests/test_clipper.py
or:
    PYTHONPATH=. python tests/test_clipper.py
"""

import sys
import shutil
from pathlib import Path

# Ensure project root is on the path regardless of how the script is invoked
sys.path.append(str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from shapely.geometry import box

from src.data.clipper import get_intersection_bounds, clip_datasets


def _make_gpkg(path: Path, minx: float, miny: float, maxx: float, maxy: float) -> None:
    """Write a single-feature GeoPackage polygon in EPSG:4326."""
    gdf = gpd.GeoDataFrame({"value": [1]}, geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")
    gdf.to_file(path, driver="GPKG")


def test_clipper():
    tmp_dir = Path("tests/temp_clipper")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    raw_dir = tmp_dir / "raw"
    raw_dir.mkdir()
    out_dir = tmp_dir / "clipped"

    try:
        # ── Create two overlapping GeoPackages ──────────────────────────────
        # File A covers lon 0–10, lat 0–10
        # File B covers lon 5–15, lat 5–15
        # Intersection: lon 5–10, lat 5–10
        file_a = raw_dir / "region_a.gpkg"
        file_b = raw_dir / "region_b.gpkg"
        _make_gpkg(file_a, 0.0, 0.0, 10.0, 10.0)
        _make_gpkg(file_b, 5.0, 5.0, 15.0, 15.0)

        files = [str(file_a), str(file_b)]

        # ── Test 1: get_intersection_bounds ─────────────────────────────────
        imin_x, imin_y, imax_x, imax_y = get_intersection_bounds(files)
        assert imin_x == 5.0,  f"Expected min_x=5.0, got {imin_x}"
        assert imin_y == 5.0,  f"Expected min_y=5.0, got {imin_y}"
        assert imax_x == 10.0, f"Expected max_x=10.0, got {imax_x}"
        assert imax_y == 10.0, f"Expected max_y=10.0, got {imax_y}"
        print("✅ Test 1 passed: get_intersection_bounds returns correct bbox.")

        # ── Test 2: ValueError when there is no overlap ─────────────────────
        no_overlap = raw_dir / "no_overlap.gpkg"
        _make_gpkg(no_overlap, 20.0, 20.0, 30.0, 30.0)
        try:
            get_intersection_bounds([str(file_a), str(no_overlap)])
            raise AssertionError("Expected ValueError for non-overlapping files.")
        except ValueError:
            print("✅ Test 2 passed: ValueError raised for non-overlapping files.")

        # ── Test 3: clip_datasets creates output files ───────────────────────
        clipped = clip_datasets(files, (imin_x, imin_y, imax_x, imax_y), str(out_dir))
        assert len(clipped) == 2, f"Expected 2 clipped files, got {len(clipped)}"

        expected_names = {"region_a_clipped.gpkg", "region_b_clipped.gpkg"}
        actual_names = {Path(p).name for p in clipped}
        assert actual_names == expected_names, f"Unexpected filenames: {actual_names}"
        print("✅ Test 3 passed: clip_datasets creates both output .gpkg files.")

        # ── Test 4: clipped bounding boxes match the intersection exactly ────
        TOLERANCE = 1e-6
        for clipped_path in clipped:
            gdf = gpd.read_file(clipped_path)
            b = gdf.total_bounds  # (min_x, min_y, max_x, max_y)
            assert abs(b[0] - imin_x) < TOLERANCE, f"{Path(clipped_path).name}: min_x {b[0]} != {imin_x}"
            assert abs(b[1] - imin_y) < TOLERANCE, f"{Path(clipped_path).name}: min_y {b[1]} != {imin_y}"
            assert abs(b[2] - imax_x) < TOLERANCE, f"{Path(clipped_path).name}: max_x {b[2]} != {imax_x}"
            assert abs(b[3] - imax_y) < TOLERANCE, f"{Path(clipped_path).name}: max_y {b[3]} != {imax_y}"
        print("✅ Test 4 passed: Clipped bounding boxes match intersection exactly.")

        print("\n🎉 All tests passed!")

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_clipper()
