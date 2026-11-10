from pathlib import Path
import json
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.distance import cdist
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from corners import anms
from harris import get_harris_corners


def extract_descriptors(gray, coords, min_std=1e-6):


    gray = np.asarray(gray, dtype=float)
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    blurred = gaussian_filter(gray, sigma=2, mode="reflect")


    yy, xx = np.meshgrid(np.arange(8) * 5 - 17.5,
                         np.arange(8) * 5 - 17.5, indexing="ij")
    h, w = gray.shape
    descs, kept = [], []
    for i, (x, y) in enumerate(coords):
        if x - 20 < 0 or y - 20 < 0 or x + 20 > w - 1 or y + 20 > h - 1:
            continue
        patch = map_coordinates(blurred, [y + yy, x + xx],
                                order=1, mode="reflect")
        sd = patch.std()
        if not np.isfinite(sd) or sd < min_std:
            continue
        descs.append(((patch - patch.mean()) / sd).ravel())
        kept.append(i)

    kept = np.asarray(kept, dtype=int)
    return coords[kept], np.asarray(descs).reshape(-1, 64), kept


def match_features(first, second, threshold=0.8):

    if len(first) == 0 or len(second) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0), np.empty(0)

    dists = cdist(first, second)
    nn = np.argsort(dists, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(first))
    d1, d2 = dists[rows, nn[:, 0]], dists[rows, nn[:, 1]]


    ratios = np.divide(d1, d2, out=np.full_like(d1, np.inf), where=d2 > 1e-12)

    accepted = np.flatnonzero(ratios < threshold)
    return (np.column_stack([accepted, nn[accepted, 0]]),
            ratios[accepted], ratios)


def examples(image, coords, descriptors, strengths, path):

    ids = np.argsort(-strengths, kind="stable")[:8]
    fig, axes = plt.subplots(3, len(ids), figsize=(16, 7))
    for col, i in enumerate(ids):
        x, y = coords[i]
        ix, iy = int(x), int(y)
        axes[0, col].imshow(image)
        axes[0, col].add_patch(Rectangle((x - 20, y - 20), 40, 40,
                                         fill=False, edgecolor="red", lw=1.5))
        axes[0, col].set_xlim(max(0, x - 75), min(image.shape[1], x + 75))
        axes[0, col].set_ylim(min(image.shape[0], y + 75), max(0, y - 75))
        axes[0, col].set_title(f"({ix}, {iy})", fontsize=9)
        axes[1, col].imshow(image[iy - 20:iy + 20, ix - 20:ix + 20])
        axes[2, col].imshow(descriptors[i].reshape(8, 8), cmap="gray",
                            vmin=-2, vmax=2, interpolation="nearest")
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    for ax, label in zip(axes[:, 0], ["Context", "40x40 source", "8x8 normalized"]):
        ax.set_ylabel(label)
    fig.suptitle("Strongest valid features (descriptor clipped to [-2, 2])")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_matches(left, right, p, q, matches, path, title):
    canvas = np.ones((max(left.shape[0], right.shape[0]),
                      left.shape[1] + right.shape[1], 3), dtype=np.uint8) * 255
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:] = right
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(canvas)
    colors = plt.cm.turbo(np.linspace(0, 1, max(len(matches), 1)))
    for color, (i, j) in zip(colors, matches):
        a = p[i]
        b = q[j] + [left.shape[1], 0]
        ax.plot([a[0], b[0]], [a[1], b[1]], "-o", color=color,
                lw=0.65, markersize=3, alpha=0.8)
    ax.set_title(f"{title}: {len(matches)} ratio-test matches (d1/d2 < 0.8)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    data = {}
    metrics = {"parameters": dict(support=40, descriptor=8, sigma=2, min_std=1e-6, ratio_threshold=0.8, distance="Euclidean", equivalent_ssd_ratio=0.64), "images": {}, "pairs": {}}

    for name in ["bank_01", "bank_03", "bank_04"]:
        image = imread(root / "images" / f"{name}.jpg")
        gray = rgb2gray(image)
        saved = root / "code" / f"{name}_corners.npz"
        if saved.exists():
            with np.load(saved) as d:
                coords = d["selected_coords"]
                strengths = d["strengths"][d["selected_indices"]]
        else:
            response, yx = get_harris_corners(gray)
            all_coords = yx.T[:, ::-1]
            vals = response[yx[0], yx[1]]
            sel, _ = anms(all_coords, vals)
            coords, strengths = all_coords[sel], vals[sel]

        valid, descs, kept = extract_descriptors(gray, coords)
        data[name] = (image, valid, descs)
        np.savez(root / "code" / f"{name}_descriptors.npz",
                 coords=valid, descriptors=descs, anms_indices=kept)
        examples(image, valid, descs, strengths[kept],
                 root / "results" / f"{name}_descriptors.png")
        metrics["images"][name] = {
            "anms_count": len(coords),
            "descriptors": len(valid),
            "discarded": len(coords) - len(valid),
        }

    for a, b in [("bank_01", "bank_03"), ("bank_03", "bank_04")]:
        image, p, d = data[a]
        other, q, e = data[b]
        matches, ratios, all_ratios = match_features(d, e)
        key = f"{a}_to_{b}"
        np.savez(root / "code" / f"{key}_matches.npz",
                 matches=matches, ratios=ratios,
                 points1=p[matches[:, 0]], points2=q[matches[:, 1]], all_ratios=all_ratios)
        plot_matches(image, other, p, q, matches,
                     root / "results" / f"{key}_matches.png",
                     f"{a} -> {b}")
        metrics["pairs"][key] = {
            "queries": len(d),
            "accepted": len(matches),
            "threshold_counts": {
                str(t): int(np.sum(all_ratios < t))
                for t in [0.6, 0.7, 0.8, 0.9]
            },
        }

    (root / "results" / "features_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
