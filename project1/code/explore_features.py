import os
import sys
import argparse
import json
import numpy as np
from skimage import io, img_as_float

from align import colorize, FEATURES

DATA_DIR = "data"
OUT_DIR = "outputs/exploration"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="e.g. emir.tif")
    parser.add_argument("--metric", default="ncc", choices=["ncc", "l2"])
    parser.add_argument("--window", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, args.filename)
    img = img_as_float(io.imread(path))

    stem = os.path.splitext(args.filename)[0]
    log = []

    for feature_name in FEATURES:
        print(f"--- feature={feature_name} metric={args.metric} ---")
        rgb, (dy_g, dx_g, score_g), (dy_r, dx_r, score_r) = colorize(
            img, use_pyramid=True, metric=args.metric, feature=feature_name,
            window=args.window,
        )
        out_path = os.path.join(OUT_DIR, f"{stem}_{feature_name}_{args.metric}.jpg")
        io.imsave(out_path, (np.clip(rgb, 0, 1) * 255).astype(np.uint8), quality=95)

        entry = {
            "feature": feature_name,
            "metric": args.metric,
            "G_offset": [dy_g, dx_g],
            "R_offset": [dy_r, dx_r],
            "G_score": round(score_g, 4),
            "R_score": round(score_r, 4),
            "output": out_path,
        }
        log.append(entry)
        print(json.dumps(entry, indent=2))

    log_path = os.path.join(OUT_DIR, f"{stem}_{args.metric}_comparison.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved {len(log)} comparison images + log to {OUT_DIR}/")
    print("Look at them side by side and eyeball which one is actually")
    print("aligned. The lowest score isn't always the best-looking result,")
    print("especially when the channels have very different brightness")
    print("curves -- the raw-pixel score gets misleading there. That gap is")
    print("worth calling out on the webpage (\"NCC picked X, but visually Y")
    print("looks better because...\").")


if __name__ == "__main__":
    main()