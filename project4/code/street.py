import numpy as np
import matplotlib.pyplot as plt
from main import pick_points, computeH, blend_images


def _load_or_pick(path, a, b, im_a, im_b, label):
    try:
        with np.load(path) as d:
            return d["pts1"], d["pts2"]
    except FileNotFoundError:
        pa = pick_points(im_a, f"{label[0]}: 12 shared features", count=12)
        pb = pick_points(im_b, f"{label[1]}: same 12, same order", count=12)
        computeH(pa, pb)
        np.savez(path, pts1=pa, pts2=pb)
        return pa, pb


if __name__ == "__main__":
    code_dir = __file__.rsplit("/", 1)[0]
    im04 = plt.imread(f"{code_dir}/../images/street_04.jpg")
    im02 = plt.imread(f"{code_dir}/../images/street_02.jpg")
    im01 = plt.imread(f"{code_dir}/../images/street_01.jpg")
    im03 = plt.imread(f"{code_dir}/../images/street_03.jpg")

    pts04, pts02 = _load_or_pick(
        f"{code_dir}/street_04_02_points.npz", "street_04", "street_02",
        im04, im02, ("Street 04", "Street 02"))
    pts02b, pts01 = _load_or_pick(
        f"{code_dir}/street_02_01_points.npz", "street_02", "street_01",
        im02, im01, ("Street 02", "Street 01"))
    pts01b, pts03 = _load_or_pick(
        f"{code_dir}/street_01_03_points.npz", "street_01", "street_03",
        im01, im03, ("Street 01", "Street 03"))

    H_04_to_02 = computeH(pts04, pts02)
    H_02_to_01 = computeH(pts02b, pts01)
    H_01_to_03 = computeH(pts01b, pts03)

    H_01_to_02 = np.linalg.inv(H_02_to_01)
    H_03_to_02 = H_01_to_02 @ np.linalg.inv(H_01_to_03)

    mosaic_four, _, _ = blend_images(
        [im04, im02, im01, im03],
        [H_04_to_02, np.eye(3), H_01_to_02, H_03_to_02])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(mosaic_four)
    ax.set_title("Street 04 + 02 + 01 + 03")
    ax.axis("off")
    plt.show()
