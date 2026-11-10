import numpy as np
import matplotlib.pyplot as plt
from main import pick_points, computeH, blend_images


if __name__ == "__main__":
    code_dir = __file__.rsplit("/", 1)[0]
    im1 = plt.imread(f"{code_dir}/../images/river_01.jpg")
    im2 = plt.imread(f"{code_dir}/../images/river_02.jpg")
    points_file = f"{code_dir}/river_01_02_points.npz"
    try:
        with np.load(points_file) as data:
            pts1, pts2 = data["pts1"], data["pts2"]
    except FileNotFoundError:
        pts1 = pick_points(im1, "River 01: 12 shared features", count=12)
        pts2 = pick_points(im2, "River 02: same 12, same order", count=12)
        computeH(pts1, pts2)
        np.savez(points_file, pts1=pts1, pts2=pts2)

    H = computeH(pts1, pts2)
    proj = (H @ np.column_stack([pts1, np.ones(len(pts1))]).T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - pts2, axis=1)
    print(f"River 01->02 mean error: {err.mean():.2f}px")

    mosaic, _, _ = blend_images([im1, im2], [H, np.eye(3)])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(mosaic)
    ax.set_title("River 01 + River 02")
    ax.axis("off")
    plt.show()
