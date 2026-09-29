"""Shared helpers for loading images, saving results, and applying filters."""

import json
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
from scipy.signal import convolve2d
from skimage.transform import resize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "images")
OUT_DIR = os.path.join(ROOT, "web", "results")
os.makedirs(OUT_DIR, exist_ok=True)


def img_path(name):
    return os.path.join(IMG_DIR, name)


def have(*names):
    missing = []
    for name in names:
        if not os.path.exists(img_path(name)):
            missing.append(name)

    if missing:
        print("  skipping - add these to images/:", ", ".join(missing))
        return False

    return True


def load(name, gray=False, max_side=700):
    img = skio.imread(img_path(name), as_gray=gray)
    img = sk.img_as_float(img)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    height, width = img.shape[:2]
    scale = max_side / max(height, width)

    if scale < 1:
        new_height = round(height * scale)
        new_width = round(width * scale)
        img = resize(img, (new_height, new_width), anti_aliasing=True)

    return img


def save(name, img, signed=False):
    img = np.asarray(img, dtype=float)

    if signed:
        max_value = np.abs(img).max() or 1.0
        img = img / (2 * max_value) + 0.5

    img = np.clip(img, 0, 1)
    path = os.path.join(OUT_DIR, name)

    if img.ndim == 2:
        plt.imsave(path, img, cmap="gray", vmin=0, vmax=1)
    else:
        image_8bit = (img * 255).round().astype(np.uint8)
        skio.imsave(path, image_8bit, check_contrast=False)

    return path


def save_json(name, data):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def gaussian_kernel(sigma, ksize=None):
    if ksize is None:
        ksize = 2 * int(np.ceil(3 * sigma)) + 1

    one_dimensional = cv2.getGaussianKernel(ksize, sigma)
    return one_dimensional @ one_dimensional.T


def conv(img, kernel, boundary="symm"):
    if img.ndim == 3:
        channels = []
        for channel in range(img.shape[2]):
            channels.append(conv(img[..., channel], kernel, boundary))
        return np.dstack(channels)

    return convolve2d(img, kernel, mode="same", boundary=boundary)


def blur(img, sigma):
    if sigma <= 0:
        return img.copy()

    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    gaussian = cv2.getGaussianKernel(ksize, sigma)

    blurred = conv(img, gaussian)
    blurred = conv(blurred, gaussian.T)
    return blurred


def to_gray(img):
    if img.ndim == 2:
        return img

    red = 0.2125 * img[..., 0]
    green = 0.7154 * img[..., 1]
    blue = 0.0721 * img[..., 2]
    return red + green + blue


def log_fft(img):
    gray = to_gray(img)
    fft = np.fft.fft2(gray)
    centered_fft = np.fft.fftshift(fft)
    return np.log(np.abs(centered_fft) + 1e-8)


def save_fft_grid(name, images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(log_fft(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=110)
    plt.close(fig)
