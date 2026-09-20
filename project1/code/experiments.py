import os
import json
import time
import numpy as np
from skimage import io, img_as_float

from align import colorize

DATA_DIR = "data"
OUT_DIR = "outputs"

OVERRIDES = {
    "emir.tif": {"feature": "sobel"},
}


DEFAULTS = dict(
    window=15,
    metric="ncc",
    feature="identity",
    border_frac=0.1,
    min_size=400,
    refine_window=4,
)

SMALL_JPGS = {"cathedral.jpg", "monastery.jpg", "tobolsk.jpg"}


def load_image(path):
    img = io.imread(path)
    return img_as_float(img)


def run_one(fname, use_pyramid, metric):
    path = os.path.join(DATA_DIR, fname)
    img = load_image(path)

    params = dict(DEFAULTS)
    params["metric"] = metric
    params.update(OVERRIDES.get(fname, {}))

    t0 = time.time()
    rgb, (dy_g, dx_g, score_g), (dy_r, dx_r, score_r) = colorize(
        img, use_pyramid=use_pyramid, **params
    )
    elapsed = time.time() - t0

    out_name = f"{os.path.splitext(fname)[0]}_{metric}.jpg"
    out_path = os.path.join(OUT_DIR, out_name)
    io.imsave(out_path, (np.clip(rgb, 0, 1) * 255).astype(np.uint8),
              quality=95)

    return {
        "file": fname,
        "mode": "pyramid" if use_pyramid else "single_scale",
        "metric": metric,
        "feature": params["feature"],
        "G_offset": [dy_g, dx_g],
        "R_offset": [dy_r, dx_r],
        "seconds": round(elapsed, 2),
        "output": out_path,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff")))

    results = []

    for fname in files:
        if fname in SMALL_JPGS:
            for metric in ("ncc", "l2"):
                print(f"[single-scale] {fname} ({metric}) ...")
                results.append(run_one(fname, use_pyramid=False, metric=metric))

    for fname in files:
        for metric in ("ncc", "l2"):
            print(f"[pyramid] {fname} ({metric}) ...")
            results.append(run_one(fname, use_pyramid=True, metric=metric))

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write("| File | Mode | Metric | Feature | G offset (dy,dx) | R offset (dy,dx) | Time (s) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['file']} | {r['mode']} | {r['metric']} | {r['feature']} "
                    f"| {tuple(r['G_offset'])} | {tuple(r['R_offset'])} | {r['seconds']} |\n")

    print(f"\nDone. {len(results)} runs logged to {OUT_DIR}/results.md and results.json")


if __name__ == "__main__":
    main()