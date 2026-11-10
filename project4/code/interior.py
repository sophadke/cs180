import numpy as np
import matplotlib.pyplot as plt
from main import pick_points, computeH, warpImageBilinear


if __name__ == "__main__":
    code_dir = __file__.rsplit("/", 1)[0]
    room = plt.imread(f"{code_dir}/../images/cafe.jpg")

    step = max(1, int(np.ceil(max(room.shape[:2]) / 1600)))
    room = room[::step, ::step]
    poster = plt.imread(f"{code_dir}/../images/river_02.jpg")
    poster_step = max(1, int(np.ceil(max(poster.shape[:2]) / 900)))
    poster = poster[::poster_step, ::poster_step]
    height, width = room.shape[:2]
    points_file = f"{code_dir}/cafe_poster_points.npz"
    try:
        with np.load(points_file) as data:
            wall_points = data["wall_points"]
            if tuple(data["room_shape"]) != room.shape:
                raise ValueError("Room working resolution changed; reselect the placement.")
        redo = input("Reselect poster position? [y/N]: ").strip().lower() == "y"
    except FileNotFoundError:
        redo = True
    if redo:
        print("Click where the poster corners should go: TL -> TR -> BR -> BL.")
        print("Choose a clear wall area; follow its perspective using slats and shelf edges.")
        wall_points = pick_points(room, "Poster position: TL -> TR -> BR -> BL", count=4)

    ph, pw = poster.shape[:2]
    poster_points = np.array([[0, 0], [pw - 1, 0],
                              [pw - 1, ph - 1], [0, ph - 1]], dtype=float)
    H = computeH(poster_points, wall_points)

    warped, mask, _ = warpImageBilinear(
        poster, H, output_bounds=(0, 0, width - 1, height - 1))
    result = room.copy()
    result[mask] = warped[mask]
    print("Poster -> wall homography:")
    print(H)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    axes[0].imshow(room)
    outline = np.vstack([wall_points, wall_points[0]])
    axes[0].plot(outline[:, 0], outline[:, 1], "r-o")
    axes[0].set_title("Cafe: selected wall position")
    axes[1].imshow(poster)
    axes[1].set_title("Sample artwork: OpenCV river photograph")
    axes[2].imshow(result)
    axes[2].set_title("Homography-based poster preview")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    if input("Save this placement and preview? [y/N]: ").strip().lower() == "y":
        np.savez(points_file, wall_points=wall_points, room_shape=room.shape, H=H)
        plt.imsave(f"{code_dir}/../results/cafe_poster.png", result)
        print("Saved cafe_poster_points.npz and results/cafe_poster.png.")
    print("This is a flat compositing preview: no automatic lighting, shadows, or occlusion.")
