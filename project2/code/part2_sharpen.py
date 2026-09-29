"""Part 2.1: image sharpening with an unsharp mask filter."""

import numpy as np

from utils import blur, conv, gaussian_kernel, have, load, save

SIGMA = 2.0
AMOUNTS = [0.5, 1, 2, 4]
DEFAULT_ALPHA = 1.5

TO_SHARPEN = ["taj.jpg", "sharpen_mine.jpg"]
SHARP_TEST = "sharp_test.jpg"


def unsharp_kernel(sigma, alpha):
    gaussian = gaussian_kernel(sigma)
    impulse = np.zeros_like(gaussian)

    center_row = gaussian.shape[0] // 2
    center_col = gaussian.shape[1] // 2
    impulse[center_row, center_col] = 1

    return (1 + alpha) * impulse - alpha * gaussian


def sharpen(img, sigma=SIGMA, alpha=DEFAULT_ALPHA):
    kernel = unsharp_kernel(sigma, alpha)
    sharpened = conv(img, kernel)
    return np.clip(sharpened, 0, 1)


def show_steps(name, img):
    stem = name.rsplit(".", 1)[0]

    low_freq = blur(img, SIGMA)
    high_freq = img - low_freq

    save(f"2_1_{stem}_original.png", img)
    save(f"2_1_{stem}_blurred.png", low_freq)
    save(f"2_1_{stem}_high.png", high_freq, signed=True)
    save(f"2_1_{stem}_sharpened.png", sharpen(img))

    for alpha in AMOUNTS:
        save(f"2_1_{stem}_alpha{alpha}.png", sharpen(img, alpha=alpha))

    two_step = np.clip(img + DEFAULT_ALPHA * high_freq, 0, 1)
    difference = np.abs(sharpen(img) - two_step).max()
    print(f"  {name}: max |single filter - two step| = {difference:.2e}")


def run():
    print("Part 2.1: sharpening")

    for name in TO_SHARPEN:
        if have(name):
            show_steps(name, load(name))

    if not have(SHARP_TEST):
        return

    img = load(SHARP_TEST)
    blurred = blur(img, SIGMA)
    resharpened = sharpen(blurred, alpha=2)

    save("2_1_eval_original.png", img)
    save("2_1_eval_blurred.png", blurred)
    save("2_1_eval_resharpened.png", resharpened)

    blurred_error = np.abs(blurred - img).mean()
    resharpened_error = np.abs(resharpened - img).mean()
    print(
        f"  eval: mean error blurred {blurred_error:.4f}, "
        f"re-sharpened {resharpened_error:.4f}"
    )


if __name__ == "__main__":
    run()
