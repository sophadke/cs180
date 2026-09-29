"""Pick two matching points on each image for hybrid image alignment."""

import json
import os
import sys

import matplotlib.pyplot as plt
import skimage.io as skio

IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "images",
)


def pick(name):
    img = skio.imread(os.path.join(IMG_DIR, name))

    plt.imshow(img)
    plt.title(f"{name}: click 2 points (e.g. left eye, right eye)")
    points = plt.ginput(2, timeout=0)
    plt.close()

    height, width = img.shape[:2]
    normalized_points = []
    for x, y in points:
        normalized_points.append([x / width, y / height])

    return normalized_points


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    image1, image2, pair_name = sys.argv[1:]
    data = {
        "im1": pick(image1),
        "im2": pick(image2),
    }

    output_path = os.path.join(IMG_DIR, f"{pair_name}_points.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print("saved", output_path)
