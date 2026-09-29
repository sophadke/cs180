"""Part 1: Fun with Filters (1.1 convolution, 1.2 finite differences, 1.3 DoG, bells & whistles)."""
import time

import numpy as np
import skimage.color
import skimage.data
from scipy.signal import convolve2d
from skimage.transform import resize

from convolution import conv_four_loops, conv_two_loops
from utils import gaussian_kernel, have, img_path, load, save, save_json
import os

D_x = np.array([[1, -1]])
D_y = np.array([[1], [-1]])

THRESH_RAW = 0.25        # edge threshold for the plain finite-difference gradient 
THRESH_DOG = 0.07        # edge threshold after Gaussian smoothing 
SIGMA = 1.5              # Gaussian 
SELFIE = "me.jpg"        # Me!


def gradient(img, dx_kernel, dy_kernel):
    dx = convolve2d(img, dx_kernel, mode="same", boundary="symm")
    dy = convolve2d(img, dy_kernel, mode="same", boundary="symm")
    return dx, dy, np.sqrt(dx ** 2 + dy ** 2)


def load_cameraman():
    if os.path.exists(img_path("cameraman.png")):
        return load("cameraman.png", gray=True)
    return skimage.data.camera() / 255.0


# ------------------------------------------------------------------ 1.1
def part_1_1():
    print("Part 1.1: convolution from scratch")
    if not have(SELFIE):
        return
    img = load(SELFIE, gray=True)
    h, w = img.shape
    img = resize(img, (200, round(200 * w / h)), anti_aliasing=True)   # keep 4 loops fast
    save("1_1_original.png", img)

    box = np.ones((9, 9)) / 81
    timings = {}
    for name, k in [("box", box), ("Dx", D_x), ("Dy", D_y)]:
        t = time.time(); four = conv_four_loops(img, k); t4 = time.time() - t
        t = time.time(); two = conv_two_loops(img, k);  t2 = time.time() - t
        t = time.time(); ref = convolve2d(img, k, mode="same", boundary="fill"); ts = time.time() - t
        match = bool(np.allclose(four, ref) and np.allclose(two, ref))
        timings[name] = {"four_loops": t4, "two_loops": t2, "scipy": ts, "matches_scipy": match}
        print(f"  {name}: four {t4:.3f}s  two {t2:.3f}s  scipy {ts:.5f}s  match={match}")
        save(f"1_1_{name}.png", two, signed=(name != "box"))
    timings["image_size"] = [int(img.shape[0]), int(img.shape[1])]
    save_json("timings.json", timings)


# ------------------------------------------------------------------ 1.2
def part_1_2(img):
    print("Part 1.2: finite difference operator")
    dx, dy, mag = gradient(img, D_x, D_y)
    save("1_2_dx.png", dx, signed=True)
    save("1_2_dy.png", dy, signed=True)
    save("1_2_mag.png", mag / mag.max())
    for t in (0.15, THRESH_RAW, 0.40):            # a few thresholds to justify the choice
        save(f"1_2_edges_{t:.2f}.png", (mag > t).astype(float))
    save("1_2_edges.png", (mag > THRESH_RAW).astype(float))


# ------------------------------------------------------------------ 1.3
def part_1_3(img):
    print("Part 1.3: derivative of Gaussian")
    G = gaussian_kernel(SIGMA)

    blurred = convolve2d(img, G, mode="same", boundary="symm")
    dx1, dy1, mag1 = gradient(blurred, D_x, D_y)
    save("1_3_blurred.png", blurred)
    save("1_3_dx.png", dx1, signed=True)
    save("1_3_dy.png", dy1, signed=True)
    save("1_3_mag.png", mag1 / mag1.max())
    save("1_3_edges.png", (mag1 > THRESH_DOG).astype(float))

    DoG_x = convolve2d(G, D_x)                    # "full" mode keeps the whole filter
    DoG_y = convolve2d(G, D_y)
    save("1_3_dog_x_filter.png", resize(DoG_x, (150, 150), order=0), signed=True)
    save("1_3_dog_y_filter.png", resize(DoG_y, (150, 150), order=0), signed=True)
    save("1_3_gaussian_filter.png", resize(G / G.max(), (150, 150), order=0))
    dx2, dy2, mag2 = gradient(img, DoG_x, DoG_y)
    save("1_3_dog_mag.png", mag2 / mag2.max())
    save("1_3_dog_edges.png", (mag2 > THRESH_DOG).astype(float))

    # Same result?
    b = G.shape[0]
    diff = np.abs(mag1 - mag2)[b:-b, b:-b].max()
    print(f"  max |two-step - one-step| (interior) = {diff:.2e}")
    return mag1, dx1, dy1, diff


# ------------------------------------------------------------------ bells & whistles
def orientation(dx, dy, mag):
    theta = np.arctan2(-dy, dx)
    hue = (theta + np.pi) / (2 * np.pi)                         
    value = np.clip(mag / np.percentile(mag, 99), 0, 1)       
    hsv = np.dstack([hue, np.ones_like(hue), value])
    save("1_bw_orientation.png", skimage.color.hsv2rgb(hsv))
    edge_only = np.dstack([hue, np.ones_like(hue), (mag > THRESH_DOG).astype(float)])
    save("1_bw_orientation_edges.png", skimage.color.hsv2rgb(edge_only))

    yy, xx = np.mgrid[-1:1:200j, -1:1:200j]
    wheel = np.dstack([(np.arctan2(-yy, xx) + np.pi) / (2 * np.pi), np.ones_like(xx),
                       (xx ** 2 + yy ** 2 <= 1).astype(float)])
    save("1_bw_wheel.png", skimage.color.hsv2rgb(wheel))


def run():
    part_1_1()
    cam = load_cameraman()
    save("1_cameraman.png", cam)
    part_1_2(cam)
    mag, dx, dy, diff = part_1_3(cam)
    orientation(dx, dy, mag)
    return {"dog_max_diff": float(diff)}


if __name__ == "__main__":
    run()
