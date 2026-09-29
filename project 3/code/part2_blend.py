"""Parts 2.3 and 2.4: Gaussian/Laplacian stacks and multiresolution blending (Burt & Adelson 1983).

No built-in pyramid functions are used: the stacks are built from our own Gaussian blur.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

from utils import OUT_DIR, blur, have, img_path, load, save

LEVELS = 6            # number of Gaussian stack levels
SIGMA0 = 2            # blur for the first level; doubles every level, like a pyramid halving size
MASK_SIGMA0 = 8       # the mask gets a wider blur so the seam fades gently


BLENDS = [
    ("oraple", "apple.jpeg", "orange.jpeg", "vertical"),
    ("blend2", "blend2_a.jpg", "blend2_b.jpg", "horizontal"),
    ("blend3", "blend3_a.jpg", "blend3_b.jpg", ("ellipse", 0.42, 0.40, 0.23, 0.24)),
]
FAVORITE = "oraple"   
FIGURE10 = ["oraple", "blend3"]   # blends that also get a Figure 10 style Laplacian grid


# ------------------------------------------------------------------ 2.3 stacks
def gaussian_stack(img, levels=LEVELS, sigma0=SIGMA0):
    stack = [img]
    for i in range(1, levels):
        stack.append(blur(stack[-1], sigma0 * 2 ** (i - 1)))   
    return stack


def laplacian_stack(img, levels=LEVELS, sigma0=SIGMA0):
    g = gaussian_stack(img, levels, sigma0)
    lap = [g[i] - g[i + 1] for i in range(levels - 1)]        
    lap.append(g[-1])                                      
    return lap                                                 


def show_stack(name, stack, signed):
    fig, axes = plt.subplots(1, len(stack), figsize=(3 * len(stack), 3.2))
    for i, (ax, im) in enumerate(zip(axes, stack)):
        if signed and i < len(stack) - 1:
            m = np.abs(im).max() or 1
            im = im / (2 * m) + 0.5
        ax.imshow(np.clip(im, 0, 1), cmap="gray" if im.ndim == 2 else None)
        ax.set_title(f"level {i}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------ 2.4 blending
def make_mask(spec, shape):
    h, w = shape[:2]
    if spec == "vertical":                     # left half
        m = np.zeros((h, w)); m[:, : w // 2] = 1
    elif spec == "horizontal":                #top half
        m = np.zeros((h, w)); m[: h // 2, :] = 1
    elif isinstance(spec, tuple) and spec[0] == "ellipse":
        _, cy, cx, ry, rx = spec
        yy, xx = np.mgrid[0:h, 0:w]
        m = (((yy - cy * h) / (ry * h)) ** 2 + ((xx - cx * w) / (rx * w)) ** 2 <= 1).astype(float)
    else:                                      # a mask image from images/
        m = load(spec, gray=True)
        m = (resize(m, (h, w)) > 0.5).astype(float)
    return m


def blend(a, b, mask):
    la, lb = laplacian_stack(a), laplacian_stack(b)
    gm = gaussian_stack(mask, sigma0=MASK_SIGMA0)
    if a.ndim == 3:
        gm = [m[..., None] for m in gm]        
    parts_a = [m * l for m, l in zip(gm, la)]
    parts_b = [(1 - m) * l for m, l in zip(gm, lb)]
    combined = [pa + pb for pa, pb in zip(parts_a, parts_b)]
    return np.clip(sum(combined), 0, 1), parts_a, parts_b, combined


def figure_342(name, parts_a, parts_b, combined):
    rows = [0, 2, 4]
    fig, axes = plt.subplots(4, 3, figsize=(10, 13))

    def show(ax, im, signed=True):
        if signed:
            m = np.abs(im).max() or 1
            im = im / (2 * m) + 0.5
        ax.imshow(np.clip(im, 0, 1))
        ax.axis("off")

    for r, lvl in enumerate(rows):
        for c, st in enumerate([parts_a, parts_b, combined]):
            show(axes[r, c], st[lvl])
            axes[r, c].set_title(f"level {lvl}")
    show(axes[3, 0], sum(parts_a), signed=False)
    show(axes[3, 1], sum(parts_b), signed=False)
    show(axes[3, 2], sum(combined), signed=False)
    for c, t in enumerate(["masked A", "masked B", "blended"]):
        axes[3, c].set_title(t)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=90)
    plt.close(fig)


def run():
    for name, fa, fb, spec in BLENDS:
        if not have(fa, fb) or (isinstance(spec, str) and "." in spec and not have(spec)):
            continue
        a = load(fa)
        b = resize(load(fb), a.shape[:2], anti_aliasing=True)
        if a.ndim != b.ndim:
            a, b = (np.dstack([x] * 3) if x.ndim == 2 else x for x in (a, b))
        mask = make_mask(spec, a.shape)
        result, pa, pb, comb = blend(a, b, mask)

        save(f"2_4_{name}_a.png", a)
        save(f"2_4_{name}_b.png", b)
        save(f"2_4_{name}_mask.png", mask)
        save(f"2_4_{name}.png", result)
        m3 = mask[..., None] if a.ndim == 3 else mask
        save(f"2_4_{name}_naive.png", m3 * a + (1 - m3) * b)

        if name in FIGURE10 and name != FAVORITE:
            figure_342(f"2_4_{name}_fig10.png", pa, pb, comb)

        if name == FAVORITE:
            show_stack(f"2_3_{name}_gauss_a.png", gaussian_stack(a), signed=False)
            show_stack(f"2_3_{name}_lap_a.png", laplacian_stack(a), signed=True)
            show_stack(f"2_3_{name}_gauss_b.png", gaussian_stack(b), signed=False)
            show_stack(f"2_3_{name}_lap_b.png", laplacian_stack(b), signed=True)
            show_stack(f"2_3_{name}_gauss_mask.png", gaussian_stack(mask, sigma0=MASK_SIGMA0), signed=False)
            figure_342(f"2_3_{name}_fig342.png", pa, pb, comb)

            # Bell & whistles
            gray = lambda x: np.dstack([x.mean(axis=2)] * 3) if x.ndim == 3 else x
            save(f"2_4_{name}_gray.png", blend(gray(a), gray(b), mask)[0])


if __name__ == "__main__":
    run()
