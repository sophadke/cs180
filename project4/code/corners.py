import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from harris import get_harris_corners


def anms(coords, strengths, count=500, c_robust=0.9):


    coords = np.asarray(coords, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    radii = np.full(len(coords), np.inf)


    for i in range(len(coords)):
        stronger = strengths * c_robust > strengths[i]
        stronger[i] = False
        if stronger.any():
            d = np.linalg.norm(coords[stronger] - coords[i], axis=1)
            radii[i] = d.min()


    order = np.lexsort((np.arange(len(coords)), -strengths, -radii))
    return order[:count], radii


def overlay(image, coords, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, facecolors="none",
               edgecolors="#ff3030", linewidths=0.65)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=["bank_01", "bank_03"])
    ap.add_argument("--count", type=int, default=500)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    metrics = {}
    for name in args.images:
        image = imread(root / "images" / f"{name}.jpg")
        response, yx = get_harris_corners(rgb2gray(image), edge_discard=20)
        coords = yx.T[:, ::-1]
        strengths = response[yx[0], yx[1]]

        selected, radii = anms(coords, strengths, args.count)
        chosen = coords[selected]

        np.savez(root / "code" / f"{name}_corners.npz",
                 coords=coords, strengths=strengths, radii=radii,
                 selected_indices=selected, selected_coords=chosen)

        pretty = name.replace("_", " ").title()
        overlay(image, coords,
                f"{pretty}: all {len(coords):,} Harris corners",
                root / "results" / f"{name}_harris.png")
        overlay(image, chosen,
                f"{pretty}: {len(chosen)} after ANMS",
                root / "results" / f"{name}_anms.png")

        metrics[name] = {
            "candidates": int(len(coords)),
            "selected": int(len(chosen)),
            "smallest_selected_radius": float(radii[selected].min()) if len(selected) else None,
            "sigma": 1, "edge_discard": 20, "c_robust": 0.9, "image_shape": list(image.shape),
        }

    (root / "results" / "corners_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
