"""Parts 2.3 and 2.4: Gaussian/Laplacian stacks and multiresolution blending."""

import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

from utils import OUT_DIR, blur, have, load, save

LEVELS = 6
SIGMA0 = 2
MASK_SIGMA0 = 8

BLENDS = [
    ("oraple", "apple.jpeg", "orange.jpeg", "vertical"),
    ("blend2", "blend2_a.jpg", "blend2_b.jpg", "horizontal"),
    ("blend3", "blend3_a.jpg", "blend3_b.jpg", ("ellipse", 0.42, 0.40, 0.23, 0.24)),
]
FAVORITE = "oraple"
FIGURE10 = ["oraple", "blend3"]


def gaussian_stack(img, levels=LEVELS, sigma0=SIGMA0):
    stack = [img]

    for level in range(1, levels):
        sigma = sigma0 * 2 ** (level - 1)
        stack.append(blur(stack[-1], sigma))

    return stack


def laplacian_stack(img, levels=LEVELS, sigma0=SIGMA0):
    gaussian = gaussian_stack(img, levels, sigma0)
    laplacian = []

    for level in range(levels - 1):
        laplacian.append(gaussian[level] - gaussian[level + 1])

    laplacian.append(gaussian[-1])
    return laplacian


def show_stack(name, stack, signed):
    fig, axes = plt.subplots(1, len(stack), figsize=(3 * len(stack), 3.2))

    for level, (ax, image) in enumerate(zip(axes, stack)):
        display = image
        if signed and level < len(stack) - 1:
            max_value = np.abs(display).max() or 1
            display = display / (2 * max_value) + 0.5

        cmap = "gray" if display.ndim == 2 else None
        ax.imshow(np.clip(display, 0, 1), cmap=cmap)
        ax.set_title(f"level {level}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=100)
    plt.close(fig)


def make_mask(spec, shape):
    height, width = shape[:2]

    if spec == "vertical":
        mask = np.zeros((height, width))
        mask[:, : width // 2] = 1
    elif spec == "horizontal":
        mask = np.zeros((height, width))
        mask[: height // 2, :] = 1
    elif isinstance(spec, tuple) and spec[0] == "ellipse":
        _, center_y, center_x, radius_y, radius_x = spec
        yy, xx = np.mgrid[0:height, 0:width]
        y_term = ((yy - center_y * height) / (radius_y * height)) ** 2
        x_term = ((xx - center_x * width) / (radius_x * width)) ** 2
        mask = (y_term + x_term <= 1).astype(float)
    else:
        mask = load(spec, gray=True)
        mask = resize(mask, (height, width))
        mask = (mask > 0.5).astype(float)

    return mask


def blend(a, b, mask):
    stack_a = laplacian_stack(a)
    stack_b = laplacian_stack(b)
    mask_stack = gaussian_stack(mask, sigma0=MASK_SIGMA0)

    if a.ndim == 3:
        mask_stack = [level[..., None] for level in mask_stack]

    parts_a = []
    parts_b = []
    combined = []

    for mask_level, level_a, level_b in zip(mask_stack, stack_a, stack_b):
        part_a = mask_level * level_a
        part_b = (1 - mask_level) * level_b
        parts_a.append(part_a)
        parts_b.append(part_b)
        combined.append(part_a + part_b)

    result = np.clip(sum(combined), 0, 1)
    return result, parts_a, parts_b, combined


def figure_342(name, parts_a, parts_b, combined):
    levels = [0, 2, 4]
    fig, axes = plt.subplots(4, 3, figsize=(10, 13))

    def show(ax, image, signed=True):
        display = image
        if signed:
            max_value = np.abs(display).max() or 1
            display = display / (2 * max_value) + 0.5

        ax.imshow(np.clip(display, 0, 1))
        ax.axis("off")

    stacks = [parts_a, parts_b, combined]
    for row, level in enumerate(levels):
        for col, stack in enumerate(stacks):
            show(axes[row, col], stack[level])
            axes[row, col].set_title(f"level {level}")

    show(axes[3, 0], sum(parts_a), signed=False)
    show(axes[3, 1], sum(parts_b), signed=False)
    show(axes[3, 2], sum(combined), signed=False)

    titles = ["masked A", "masked B", "blended"]
    for col, title in enumerate(titles):
        axes[3, col].set_title(title)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=90)
    plt.close(fig)


def run():
    print("Parts 2.3 + 2.4: stacks and blending")

    for name, file_a, file_b, mask_spec in BLENDS:
        if not have(file_a, file_b):
            continue
        if isinstance(mask_spec, str) and "." in mask_spec and not have(mask_spec):
            continue

        image_a = load(file_a)
        image_b = resize(load(file_b), image_a.shape[:2], anti_aliasing=True)

        if image_a.ndim != image_b.ndim:
            if image_a.ndim == 2:
                image_a = np.dstack([image_a, image_a, image_a])
            if image_b.ndim == 2:
                image_b = np.dstack([image_b, image_b, image_b])

        mask = make_mask(mask_spec, image_a.shape)
        result, parts_a, parts_b, combined = blend(image_a, image_b, mask)

        save(f"2_4_{name}_a.png", image_a)
        save(f"2_4_{name}_b.png", image_b)
        save(f"2_4_{name}_mask.png", mask)
        save(f"2_4_{name}.png", result)

        mask_for_image = mask[..., None] if image_a.ndim == 3 else mask
        naive = mask_for_image * image_a + (1 - mask_for_image) * image_b
        save(f"2_4_{name}_naive.png", naive)

        if name in FIGURE10 and name != FAVORITE:
            figure_342(f"2_4_{name}_fig10.png", parts_a, parts_b, combined)

        if name == FAVORITE:
            show_stack(f"2_3_{name}_gauss_a.png", gaussian_stack(image_a), signed=False)
            show_stack(f"2_3_{name}_lap_a.png", laplacian_stack(image_a), signed=True)
            show_stack(f"2_3_{name}_gauss_b.png", gaussian_stack(image_b), signed=False)
            show_stack(f"2_3_{name}_lap_b.png", laplacian_stack(image_b), signed=True)
            show_stack(
                f"2_3_{name}_gauss_mask.png",
                gaussian_stack(mask, sigma0=MASK_SIGMA0),
                signed=False,
            )
            figure_342(f"2_3_{name}_fig342.png", parts_a, parts_b, combined)

            gray_a = image_a
            gray_b = image_b
            if image_a.ndim == 3:
                mean_a = image_a.mean(axis=2)
                mean_b = image_b.mean(axis=2)
                gray_a = np.dstack([mean_a, mean_a, mean_a])
                gray_b = np.dstack([mean_b, mean_b, mean_b])

            gray_result = blend(gray_a, gray_b, mask)[0]
            save(f"2_4_{name}_gray.png", gray_result)


if __name__ == "__main__":
    run()
