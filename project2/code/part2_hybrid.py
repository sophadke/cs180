"""Part 2.2: hybrid images."""

import json
import os

import numpy as np
from skimage.transform import SimilarityTransform, resize, warp

from utils import blur, have, img_path, load, save, save_fft_grid, save_json, to_gray

PAIRS = [
    ("derek_nutmeg", "DerekPicture.jpg", "nutmeg.jpg", 6, 3),
    ("pair2", "pair2_far.jpg", "pair2_close.jpg", 6, 3),
    ("pair3", "pair3_far.jpg", "pair3_close.jpg", 6, 3),
]
FAVORITE = "derek_nutmeg"


def align(pair, im1, im2):
    points_path = img_path(f"{pair}_points.json")

    if not os.path.exists(points_path):
        print(f"  no {pair}_points.json - run pick_points.py for better alignment; resizing only")
        resized = resize(im2, im1.shape[:2], anti_aliasing=True)
        return im1, resized

    with open(points_path) as f:
        points = json.load(f)

    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]

    points1 = np.array([
        [x * width1, y * height1]
        for x, y in points["im1"]
    ])
    points2 = np.array([
        [x * width2, y * height2]
        for x, y in points["im2"]
    ])

    transform = SimilarityTransform()
    transform.estimate(points1, points2)

    aligned_im2 = warp(
        im2,
        transform,
        output_shape=im1.shape[:2],
        mode="edge",
    )
    return im1, aligned_im2


def hybrid(im_far, im_close, sigma_low, sigma_high):
    low_freq = blur(im_far, sigma_low)
    high_freq = im_close - blur(im_close, sigma_high)
    combined = np.clip(low_freq + high_freq, 0, 1)
    return low_freq, high_freq, combined


def match_channels(a, b):
    if a.ndim == b.ndim:
        return a, b

    if a.ndim == 2:
        a = np.dstack([a, a, a])
    if b.ndim == 2:
        b = np.dstack([b, b, b])

    return a, b


def cutoff_frequency(sigma):
    numerator = np.sqrt(2 * np.log(2))
    denominator = 2 * np.pi * sigma
    return round(float(numerator / denominator), 4)


def run():
    print("Part 2.2: hybrid images")
    info = {}

    for pair, far_name, close_name, sigma_low, sigma_high in PAIRS:
        if not have(far_name, close_name):
            continue

        far_img = load(far_name)
        close_img = load(close_name)
        far_img, close_img = align(pair, far_img, close_img)

        far_gray = to_gray(far_img)
        close_gray = to_gray(close_img)
        low_freq, high_freq, hybrid_img = hybrid(
            far_gray,
            close_gray,
            sigma_low,
            sigma_high,
        )

        save(f"2_2_{pair}_far.png", far_img)
        save(f"2_2_{pair}_close.png", close_img)
        save(f"2_2_{pair}_hybrid.png", hybrid_img)

        small_shape = (hybrid_img.shape[0] // 4, hybrid_img.shape[1] // 4)
        small_hybrid = resize(hybrid_img, small_shape, anti_aliasing=True)
        save(f"2_2_{pair}_hybrid_small.png", small_hybrid)

        info[pair] = {
            "low_cutoff_cyc_per_px": cutoff_frequency(sigma_low),
            "high_cutoff_cyc_per_px": cutoff_frequency(sigma_high),
            "sigma_low": sigma_low,
            "sigma_high": sigma_high,
        }

        if pair != FAVORITE:
            continue

        save(f"2_2_{pair}_low.png", low_freq)
        save(f"2_2_{pair}_high.png", high_freq, signed=True)
        save_fft_grid(
            f"2_2_{pair}_fft.png",
            [far_gray, close_gray, low_freq, high_freq, hybrid_img],
            ["far image", "close image", "low-passed", "high-passed", "hybrid"],
        )

        color_tests = [
            ("color_low", far_img, close_gray),
            ("color_high", far_gray, close_img),
            ("color_both", far_img, close_img),
        ]

        for tag, first, second in color_tests:
            first, second = match_channels(first, second)
            _, _, color_hybrid = hybrid(first, second, sigma_low, sigma_high)
            save(f"2_2_{pair}_{tag}.png", color_hybrid)

    save_json("hybrid_cutoffs.json", info)


if __name__ == "__main__":
    run()
