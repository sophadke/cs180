import numpy as np
import matplotlib.pyplot as plt
from main import pick_points, computeH, warpImageNearestNeighbor, warpImageBilinear


if __name__ == "__main__":
    code_dir = __file__.rsplit("/", 1)[0]


    examples = [("notebook",600, 825), ("sudoku", 600, 600)]
    for name, width, height in examples:
        image = plt.imread(f"{code_dir}/../images/rectify_{name}.jpg")
        points_file = f"{code_dir}/rectify_{name}_points.npz"
        try:
            with np.load(points_file) as data:
                source_points = data["source_points"]
        except FileNotFoundError:
            print(f"{name}: select TOP LEFT, TOP RIGHT, BOTTOM RIGHT, BOTTOM LEFT.")
            if name == "notebook":
                print("Use the TOP COVER corners, not the bottom of the page block.")
                print("For rounded corners, choose where the straight edges would intersect.")
            else:
                print("Select the four corners of the OUTER Sudoku grid, not the paper.")
            source_points = pick_points(
                image, f"{name}: TL -> TR -> BR -> BL (then Enter)", count=4)
            np.savez(points_file, source_points=source_points)
            print(f"Saved {points_file}")

        target_points = np.array([[0, 0], [width - 1, 0],
                                  [width - 1, height - 1], [0, height - 1]], dtype=float)
        H = computeH(source_points, target_points)
        bounds = (0, 0, width - 1, height - 1)
        nearest, mask_nn, _ = warpImageNearestNeighbor(image, H, output_bounds=bounds)
        bilinear, mask_bil, _ = warpImageBilinear(image, H, output_bounds=bounds)
        print(f"{name} H (photo -> rectangle):")
        print(H)
        print(f"Output size: {width} x {height}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 7))
        axes[0].imshow(image)
        outline = np.vstack([source_points, source_points[0]])
        axes[0].plot(outline[:, 0], outline[:, 1], "r-o")
        for number, (x, y) in enumerate(source_points, 1):
            axes[0].annotate(str(number), (x, y), color="yellow",
                             xytext=(5, 5), textcoords="offset points")
        axes[0].set_title("Original and selected corners")
        axes[1].imshow(nearest, interpolation="nearest")
        axes[1].set_title("Rectified: nearest neighbor")
        axes[2].imshow(bilinear, interpolation="nearest")
        axes[2].set_title("Rectified: bilinear")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{name.capitalize()} rectification")
        fig.tight_layout()
        plt.show()
