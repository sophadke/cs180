"""Click alignment points for a hybrid-image pair and save them to images/<pair>_points.json.

Usage:  python3 pick_points.py DerekPicture.jpg nutmeg.jpg derek_nutmeg

A window opens for each image. Click the same two features on both
(for faces: left eye, then right eye), and the window closes by itself.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import skimage.io as skio

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")


def pick(name):
    img = skio.imread(os.path.join(IMG_DIR, name))
    plt.imshow(img)
    plt.title(f"{name}: click 2 points (e.g. left eye, right eye)")
    pts = plt.ginput(2, timeout=0)
    plt.close()
    h, w = img.shape[:2]
    # store as fractions of width/height so they still work after resizing
    return [[x / w, y / h] for x, y in pts]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    im1, im2, pair = sys.argv[1:]
    data = {"im1": pick(im1), "im2": pick(im2)}
    out = os.path.join(IMG_DIR, f"{pair}_points.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print("saved", out)
