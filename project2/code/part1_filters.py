"""Part 1: convolution, finite differences, and derivative of Gaussian filters."""

import os
import time

import numpy as np
import skimage.color
import skimage.data
from scipy.signal import convolve2d
from skimage.transform import resize

from convolution import conv_four_loops, conv_two_loops
from utils import gaussian_kernel, have, img_path, load, save, save_json

D_x = np.array([[1, -1]])
D_y = np.array([[1], [-1]])

THRESH_RAW = 0.25
THRESH_DOG = 0.07
SIGMA = 1.5
SELFIE = "me.jpg"


def gradient(img, dx_kernel, dy_kernel):
    dx = convolve2d(img, dx_kernel, mode="same", boundary="symm")
    dy = convolve2d(img, dy_kernel, mode="same", boundary="symm")
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    return dx, dy, magnitude


def load_cameraman():
    if os.path.exists(img_path("cameraman.png")):
        return load("cameraman.png", gray=True)

    return skimage.data.camera() / 255.0


def part_1_1():
    print("Part 1.1: convolution from scratch")
    if not have(SELFIE):
        return

    img = load(SELFIE, gray=True)
    height, width = img.shape
    new_width = round(200 * width / height)
    img = resize(img, (200, new_width), anti_aliasing=True)
    save("1_1_original.png", img)

    box_filter = np.ones((9, 9)) / 81
    filters = [("box", box_filter), ("Dx", D_x), ("Dy", D_y)]
    timings = {}

    for name, kernel in filters:
        start = time.time()
        four_loop_result = conv_four_loops(img, kernel)
        four_loop_time = time.time() - start

        start = time.time()
        two_loop_result = conv_two_loops(img, kernel)
        two_loop_time = time.time() - start

        start = time.time()
        scipy_result = convolve2d(img, kernel, mode="same", boundary="fill")
        scipy_time = time.time() - start

        matches = bool(
            np.allclose(four_loop_result, scipy_result)
            and np.allclose(two_loop_result, scipy_result)
        )

        timings[name] = {
            "four_loops": four_loop_time,
            "two_loops": two_loop_time,
            "scipy": scipy_time,
            "matches_scipy": matches,
        }

        print(
            f"  {name}: four {four_loop_time:.3f}s  "
            f"two {two_loop_time:.3f}s  scipy {scipy_time:.5f}s  match={matches}"
        )
        save(f"1_1_{name}.png", two_loop_result, signed=(name != "box"))

    timings["image_size"] = [int(img.shape[0]), int(img.shape[1])]
    save_json("timings.json", timings)


def part_1_2(img):
    print("Part 1.2: finite difference operator")
    dx, dy, magnitude = gradient(img, D_x, D_y)

    save("1_2_dx.png", dx, signed=True)
    save("1_2_dy.png", dy, signed=True)
    save("1_2_mag.png", magnitude / magnitude.max())

    for threshold in (0.15, THRESH_RAW, 0.40):
        edges = (magnitude > threshold).astype(float)
        save(f"1_2_edges_{threshold:.2f}.png", edges)

    save("1_2_edges.png", (magnitude > THRESH_RAW).astype(float))


def part_1_3(img):
    print("Part 1.3: derivative of Gaussian")
    gaussian = gaussian_kernel(SIGMA)

    blurred = convolve2d(img, gaussian, mode="same", boundary="symm")
    dx_blur, dy_blur, mag_blur = gradient(blurred, D_x, D_y)

    save("1_3_blurred.png", blurred)
    save("1_3_dx.png", dx_blur, signed=True)
    save("1_3_dy.png", dy_blur, signed=True)
    save("1_3_mag.png", mag_blur / mag_blur.max())
    save("1_3_edges.png", (mag_blur > THRESH_DOG).astype(float))

    dog_x = convolve2d(gaussian, D_x)
    dog_y = convolve2d(gaussian, D_y)

    save("1_3_dog_x_filter.png", resize(dog_x, (150, 150), order=0), signed=True)
    save("1_3_dog_y_filter.png", resize(dog_y, (150, 150), order=0), signed=True)
    save("1_3_gaussian_filter.png", resize(gaussian / gaussian.max(), (150, 150), order=0))

    _, _, mag_dog = gradient(img, dog_x, dog_y)
    save("1_3_dog_mag.png", mag_dog / mag_dog.max())
    save("1_3_dog_edges.png", (mag_dog > THRESH_DOG).astype(float))

    border = gaussian.shape[0]
    difference = np.abs(mag_blur - mag_dog)[border:-border, border:-border].max()
    print(f"  max |two-step - one-step| (interior) = {difference:.2e}")

    return mag_blur, dx_blur, dy_blur, difference


def orientation(dx, dy, mag):
    theta = np.arctan2(-dy, dx)
    hue = (theta + np.pi) / (2 * np.pi)
    value = np.clip(mag / np.percentile(mag, 99), 0, 1)

    hsv = np.dstack([hue, np.ones_like(hue), value])
    save("1_bw_orientation.png", skimage.color.hsv2rgb(hsv))

    edge_value = (mag > THRESH_DOG).astype(float)
    edge_hsv = np.dstack([hue, np.ones_like(hue), edge_value])
    save("1_bw_orientation_edges.png", skimage.color.hsv2rgb(edge_hsv))

    yy, xx = np.mgrid[-1:1:200j, -1:1:200j]
    wheel_hue = (np.arctan2(-yy, xx) + np.pi) / (2 * np.pi)
    wheel_value = (xx ** 2 + yy ** 2 <= 1).astype(float)
    wheel = np.dstack([wheel_hue, np.ones_like(xx), wheel_value])
    save("1_bw_wheel.png", skimage.color.hsv2rgb(wheel))


def run():
    part_1_1()

    cameraman = load_cameraman()
    save("1_cameraman.png", cameraman)
    part_1_2(cameraman)

    magnitude, dx, dy, difference = part_1_3(cameraman)
    orientation(dx, dy, magnitude)

    return {"dog_max_diff": float(difference)}


if __name__ == "__main__":
    run()
