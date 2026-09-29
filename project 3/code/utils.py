import os
import json

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                     
import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
from skimage.transform import resize
from scipy.signal import convolve2d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "images")
OUT_DIR = os.path.join(ROOT, "web", "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- files
def img_path(name):
    return os.path.join(IMG_DIR, name)


def have(*names):
    missing = [n for n in names if not os.path.exists(img_path(n))]
    if missing:
        print("  skipping - add these to images/:", ", ".join(missing))
    return not missing


def load(name, gray=False, max_side=700):
    """Read an image from images/ as floats in [0, 1], shrinking it if it is large."""
    img = skio.imread(img_path(name), as_gray=gray)
    img = sk.img_as_float(img)
    if img.ndim == 3 and img.shape[2] == 4:          # drop alpha channel
        img = img[..., :3]
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = resize(img, (round(h * scale), round(w * scale)), anti_aliasing=True)
    return img


def save(name, img, signed=False):
    """Save to web/results/. signed=True maps 0 to middle gray (for derivatives/Laplacians)."""
    img = np.asarray(img, dtype=float)
    if signed:
        m = np.abs(img).max() or 1.0
        img = img / (2 * m) + 0.5
    img = np.clip(img, 0, 1)
    path = os.path.join(OUT_DIR, name)
    if img.ndim == 2:
        plt.imsave(path, img, cmap="gray", vmin=0, vmax=1)
    else:
        skio.imsave(path, (img * 255).round().astype(np.uint8), check_contrast=False)
    return path


def save_json(name, data):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------- filters
def gaussian_kernel(sigma, ksize=None):
    if ksize is None:
        ksize = 2 * int(np.ceil(3 * sigma)) + 1       # cover +-3 sigma
    g = cv2.getGaussianKernel(ksize, sigma)           # column vector, sums to 1
    return g @ g.T


def conv(img, kernel, boundary="symm"):
    if img.ndim == 3:
        return np.dstack([conv(img[..., c], kernel, boundary) for c in range(img.shape[2])])
    return convolve2d(img, kernel, mode="same", boundary=boundary)


def blur(img, sigma):
    if sigma <= 0:
        return img.copy()
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    return conv(conv(img, g), g.T)


def to_gray(img):
    # Same weights as skimage's rgb2gray, as a plain weighted sum
    # (avoids spurious numpy matmul warnings on Apple Silicon)
    if img.ndim == 2:
        return img
    return 0.2125 * img[..., 0] + 0.7154 * img[..., 1] + 0.0721 * img[..., 2]


def log_fft(img):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(to_gray(img)))) + 1e-8)


def save_fft_grid(name, images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(log_fft(im), cmap="gray")
        ax.set_title(t)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=110)
    plt.close(fig)
