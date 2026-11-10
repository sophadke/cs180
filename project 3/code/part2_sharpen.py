"""Part 2.1: image sharpening with the unsharp mask filter."""
import numpy as np

from utils import blur, conv, gaussian_kernel, have, load, save

SIGMA = 2.0
AMOUNTS = [0.5, 1, 2, 4]         
DEFAULT_ALPHA = 1.5

TO_SHARPEN = ["taj.jpg", "sharpen_mine.jpg"]
SHARP_TEST = "sharp_test.jpg"


def unsharp_kernel(sigma, alpha):
    """(1 + alpha) * impulse - alpha * Gaussian: one filter that adds back alpha x the high frequencies."""
    G = gaussian_kernel(sigma)
    impulse = np.zeros_like(G)
    impulse[G.shape[0] // 2, G.shape[1] // 2] = 1
    return (1 + alpha) * impulse - alpha * G


def sharpen(img, sigma=SIGMA, alpha=DEFAULT_ALPHA):
    return np.clip(conv(img, unsharp_kernel(sigma, alpha)), 0, 1)


def show_steps(name, img):
    stem = name.rsplit(".", 1)[0]
    low = blur(img, SIGMA)
    high = img - low
    save(f"2_1_{stem}_original.png", img)
    save(f"2_1_{stem}_blurred.png", low)
    save(f"2_1_{stem}_high.png", high, signed=True)
    save(f"2_1_{stem}_sharpened.png", sharpen(img))
    for a in AMOUNTS:
        save(f"2_1_{stem}_alpha{a}.png", sharpen(img, alpha=a))
    two_step = np.clip(img + DEFAULT_ALPHA * high, 0, 1)
    print(f"  {name}: max |single filter - two step| = {np.abs(sharpen(img) - two_step).max():.2e}")


def run():
    print("Part 2.1: sharpening")
    for name in TO_SHARPEN:
        if have(name):
            show_steps(name, load(name))

    if have(SHARP_TEST):
        img = load(SHARP_TEST)
        blurred = blur(img, SIGMA)
        resharp = sharpen(blurred, alpha=2)
        save("2_1_eval_original.png", img)
        save("2_1_eval_blurred.png", blurred)
        save("2_1_eval_resharpened.png", resharp)
        err_b = np.abs(blurred - img).mean()
        err_s = np.abs(resharp - img).mean()
        print(f"  eval: mean error blurred {err_b:.4f}, re-sharpened {err_s:.4f}")


if __name__ == "__main__":
    run()
