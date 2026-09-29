"""Part 2.2: hybrid images (Oliva, Torralba & Schyns 2006)."""
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
    """Warp im2 so its two clicked points land on im1's two clicked points."""
    pts_file = img_path(f"{pair}_points.json")
    if not os.path.exists(pts_file):
        print(f"  no {pair}_points.json - run pick_points.py for better alignment; resizing only")
        return im1, resize(im2, im1.shape[:2], anti_aliasing=True)
    with open(pts_file) as f:
        pts = json.load(f)
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    p1 = np.array([[x * w1, y * h1] for x, y in pts["im1"]])
    p2 = np.array([[x * w2, y * h2] for x, y in pts["im2"]])
    tform = SimilarityTransform()
    tform.estimate(p1, p2)                       # maps im1 coords -> im2 coords
    im2_aligned = warp(im2, tform, output_shape=im1.shape[:2], mode="edge")
    return im1, im2_aligned


def hybrid(im_far, im_close, sigma_low, sigma_high):
    low = blur(im_far, sigma_low)                       
    high = im_close - blur(im_close, sigma_high)         
    return low, high, np.clip(low + high, 0, 1)


def match_channels(a, b):
    """Make a and b both color or both gray so they can be added."""
    if a.ndim == b.ndim:
        return a, b
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    if b.ndim == 2:
        b = np.dstack([b] * 3)
    return a, b


def run():
    print("Part 2.2: hybrid images")
    info = {}
    for pair, far, close, s_low, s_high in PAIRS:
        if not have(far, close):
            continue
        im1, im2 = align(pair, load(far), load(close))
        g1, g2 = to_gray(im1), to_gray(im2)
        low, high, hyb = hybrid(g1, g2, s_low, s_high)

        save(f"2_2_{pair}_far.png", im1)
        save(f"2_2_{pair}_close.png", im2)
        save(f"2_2_{pair}_hybrid.png", hyb)
        save(f"2_2_{pair}_hybrid_small.png", resize(hyb, (hyb.shape[0] // 4, hyb.shape[1] // 4),
                                                   anti_aliasing=True))


        info[pair] = {k: round(float(np.sqrt(2 * np.log(2)) / (2 * np.pi * s)), 4)
                      for k, s in [("low_cutoff_cyc_per_px", s_low), ("high_cutoff_cyc_per_px", s_high)]}
        info[pair].update(sigma_low=s_low, sigma_high=s_high)

        if pair == FAVORITE:
            save(f"2_2_{pair}_low.png", low)
            save(f"2_2_{pair}_high.png", high, signed=True)
            save_fft_grid(f"2_2_{pair}_fft.png", [g1, g2, low, high, hyb],
                          ["far image", "close image", "low-passed", "high-passed", "hybrid"])

            for tag, a, b in [("color_low", im1, g2), ("color_high", g1, im2), ("color_both", im1, im2)]:
                a, b = match_channels(a, b)
                _, _, h = hybrid(a, b, s_low, s_high)
                save(f"2_2_{pair}_{tag}.png", h)
    save_json("hybrid_cutoffs.json", info)


if __name__ == "__main__":
    run()
