import numpy as np
import matplotlib.pyplot as plt
from main import pick_points, computeH, warpImageBilinear


def rgb8(image):
    image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
    return image


if __name__ == "__main__":
    code_dir = __file__.rsplit("/", 1)[0]
    room = rgb8(plt.imread(f"{code_dir}/../images/cafe.jpg"))
    step = max(1, int(np.ceil(max(room.shape[:2]) / 1600)))
    room = room[::step, ::step]
    notebook = rgb8(plt.imread(f"{code_dir}/../results/rectify_notebook_bilinear.png"))

    notebook = np.rot90(notebook)
    with np.load(f"{code_dir}/cafe_wall_points.npz") as data:
        H_wall = data["H"]
        fw, fh = int(data["output_width"]), int(data["output_height"])
    placement_file = f"{code_dir}/cafe_notebook_points.npz"
    try:
        with np.load(placement_file) as data:
            points = data["points"]
            if tuple(data["room_shape"]) != room.shape:
                raise ValueError("Working image size has changed; reselect the placement.")
        redo = input("Reselect notebook placement? [y/N]: ").strip().lower() == "y"
    except FileNotFoundError:
        redo = True
    if redo:
        print("Outline the visible COVER, with its long bottom edge resting on the shelf.")
        print("Click TL -> TR -> BR -> BL; the cover should be wider than tall.")
        print("Choose inside the shelf area you previously rectified to see it in both views.")
        points = pick_points(room, "Notebook cover on shelf: TL -> TR -> BR -> BL", count=4)
    nh, nw = notebook.shape[:2]
    corners = np.array([[0, 0], [nw - 1, 0], [nw - 1, nh - 1], [0, nh - 1]], dtype=float)
    H_place = computeH(corners, points)
    height, width = room.shape[:2]
    warped, mask, _ = warpImageBilinear(
        notebook, H_place, output_bounds=(0, 0, width - 1, height - 1))
    design = room.copy()
    design[mask] = warped[mask]
    bounds = (0, 0, fw - 1, fh - 1)
    front_room, _, _ = warpImageBilinear(room, H_wall, output_bounds=bounds)

    H_front = H_wall @ H_place
    front_cover, front_mask, _ = warpImageBilinear(notebook, H_front, output_bounds=bounds)
    front_design = front_room.copy()
    front_design[front_mask] = front_cover[front_mask]
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    for ax, image, title in zip(axes, [design, front_room, front_design],
            ["Notebook on shelf: angled view", "Straightened shelf: original", "Straightened shelf: notebook preview"]):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    if input("Save this notebook placement? [y/N]: ").strip().lower() == "y":
        np.savez(placement_file, points=points, room_shape=room.shape,
                 H_place=H_place, H_wall=H_wall, H_front=H_front)
        plt.imsave(f"{code_dir}/../results/cafe_notebook_angled.png", design)
        plt.imsave(f"{code_dir}/../results/cafe_notebook_front.png", front_design)
        fig.savefig(f"{code_dir}/../results/cafe_notebook_comparison.png", dpi=140, bbox_inches="tight")
        print("Saved notebook placement and comparison images.")
    print("This represents the flat cover only; no thickness, stack, shadows, or automatic occlusion.")
