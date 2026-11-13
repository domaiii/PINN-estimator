import argparse
import os
from pathlib import Path

import numpy as np
import trimesh
import yaml
from PIL import Image
from shapely import contains_xy, union_all


def main():
    parser = argparse.ArgumentParser(description="Create a 2D occupancy map from an STL slice.")
    parser.add_argument("--file", type=Path, required=True, help="Input STL file (meters, Z up).")
    parser.add_argument("--resolution", type=float, default=0.01, help="Meters per pixel (default: 0.01).")
    parser.add_argument("--setupname", type=str, default=None, help="Setup name (default: STL file stem).")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory (default: STL parent directory).")
    parser.add_argument("--slice_height", type=float, required=True, help="Slice height in meters.")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.file.parent
    if args.setupname is None:
        args.setupname = args.file.stem
    if not np.isfinite(args.resolution) or args.resolution <= 0:
        parser.error("Resolution must be finite and positive")
    if not np.isfinite(args.slice_height):
        parser.error("Slice height must be finite")

    mesh = trimesh.load_mesh(args.file)
    slice = mesh.section(plane_origin=[0.0, 0.0, args.slice_height], plane_normal=[0.0, 0.0, 1.0])

    if slice is None:
        parser.error("No mesh section at the requested height")

    bounds = mesh.bounds[:, :2]

    transform = np.eye(4)
    transform[2, 3] = -args.slice_height

    slice_2D, _ = slice.to_2D(to_2D=transform)
    slice_2D.explode()

    pixel_size = args.resolution

    polygons = union_all(slice_2D.polygons_full)

    width, height = np.rint(
        (bounds[1] - bounds[0]) / pixel_size
    ).astype(int)

    x = bounds[0, 0] + (np.arange(width) + 0.5) * pixel_size
    y = bounds[0, 1] + (np.arange(height) + 0.5) * pixel_size
    xx, yy = np.meshgrid(x, y)

    img_array = contains_xy(polygons, xx, yy)
    img = Image.fromarray(img_array)

    img_array = np.array(img, dtype=np.uint8)
    img_array = 1 - img_array

    # Write PGM file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "occupancy.pgm").open("w") as f:
        f.write("P2\n")
        f.write(f"{img_array.shape[1]} {img_array.shape[0]}\n")
        f.write("1\n")

        for row in reversed(img_array):
            f.writelines(f"{pixel} " for pixel in row)
            f.write("\n")

    # Write YAML file
    map_info = {
        "image": "occupancy.pgm",
        "resolution": float(pixel_size),
        "origin": [float(bounds[0, 0]), float(bounds[0, 1]), 0.0],
        "negate": 0,
        "occupied_thresh": 0.9,
        "free_thresh": 0.1,
        "metadata": {
            "setup": args.setupname,
            "slice_height": args.slice_height,
            "geometry": os.path.relpath(args.file.resolve(), args.output_dir.resolve()),
        },
    }

    with (args.output_dir / "occupancy.yaml").open("w") as f:
        yaml.safe_dump(map_info, f, sort_keys=False)


if __name__ == "__main__":
    main()
