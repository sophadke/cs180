import numpy as np
import matplotlib.pyplot as plt
from main import pick_points, computeH, warpImageBilinear


if __name__ == "__main__":
    code_dir = __file__.rsplit("/", 1)[0]
    room = plt.imread(f"{code_dir}/../images/cafe.jpg")
    step = max(1, int(np.ceil(max(room.shape[:2]) / 1600)))
    room = room[::step, ::step]
    design = plt.imread(f"{code_dir}/../results/cafe_poster.png")
    if np.issubdtype(design.dtype, np.floating):
        design = np.clip(np.rint(design[:, :, :3] * 255), 0, 255).astype(np.uint8)
    if design.shape != room.shape:
        raise ValueError("Poster preview and room must have matching working resolutions.")
    points_file = f"{code_dir}/cafe_wall_points.npz"
    try:
        with np.load(points_file) as data:
            wall_points = data["wall_points"]
            ratio = float(data["width_height_ratio"])
        redo = input("Reselect wall rectangle and proportions? [y/N]: ").strip().lower() == "y"
    except FileNotFoundError:
        redo = True
    if redo:
        print("Choose four points that represent a rectangle ON THE SAME WALL PLANE.")
        print("Click TL -> TR -> BR -> BL, then Enter.")
        print("Use wall slats for vertical directions, and wall-parallel shelf edges for horizontal directions.")
        print("Use the wall behind the shelves; do not mix corners from projecting shelf fronts or the tiled side wall.")
        print("Include the poster area in your selection. A subregion of the wooden wall is sufficient.")
        wall_points = pick_points(room, "Wall rectangle: TL -> TR -> BR -> BL", count=4)
        text = input("Assumed physical width / height of selected wall area [1.5]: ").strip()
        ratio = float(text) if text else 1.5
        if not np.isfinite(ratio) or not 0.2 <= ratio <= 5:
            raise ValueError("Enter a positive width/height ratio between 0.2 and 5.")

    width = 1000
    height = int(round(width / ratio))
    target = np.array([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]], dtype=float)
    H = computeH(wall_points, target)
    bounds = (0, 0, width - 1, height - 1)
    straight_room, _, _ = warpImageBilinear(room, H, output_bounds=bounds)
    straight_design, _, _ = warpImageBilinear(design, H, output_bounds=bounds)
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    axes[0].imshow(design)
    outline = np.vstack([wall_points, wall_points[0]])
    axes[0].plot(outline[:, 0], outline[:, 1], "r-o")
    axes[0].set_title("Angled view: chosen wall plane")
    axes[1].imshow(straight_room)
    axes[1].set_title("Front-facing wall: original")
    axes[2].imshow(straight_design)
    axes[2].set_title("Front-facing wall: poster preview")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Planar rectification — assumed wall width/height = {ratio:g}")
    fig.tight_layout()
    plt.show()
    if input("Save this wall rectification? [y/N]: ").strip().lower() == "y":
        np.savez(points_file, wall_points=wall_points, width_height_ratio=ratio,
                 H=H, output_width=width, output_height=height)
        plt.imsave(f"{code_dir}/../results/cafe_wall_front.png", straight_room)
        plt.imsave(f"{code_dir}/../results/cafe_design_front.png", straight_design)
        fig.savefig(f"{code_dir}/../results/cafe_wall_comparison.png", dpi=140, bbox_inches="tight")
        print("Saved wall corners, assumed proportions, and front-facing results.")
    print("Only the selected wall plane is rectified. Projecting objects can remain distorted.")
