import numpy as np
import matplotlib.pyplot as plt


def pick_points(image, name, count=8):
    points = []
    fig, ax = plt.subplots(figsize=(10, 8))

    def redraw():
        ax.clear()
        ax.imshow(image)
        for n, (x, y) in enumerate(points, 1):
            ax.plot(x, y, "r+", markersize=12)
            ax.annotate(str(n), (x, y), xytext=(6, 6),
                        textcoords="offset points", color="yellow",
                        bbox=dict(facecolor="black", alpha=0.6))
        ax.set_title(f"{name}: {len(points)}/{count}\n"
                     "Left click: add | Right click: undo | Enter: confirm")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or fig.canvas.toolbar.mode:
            return
        if event.button == 1 and len(points) < count:
            points.append((event.xdata, event.ydata))
            print(f"{name}: point {len(points)} = {points[-1]}", flush=True)
        elif event.button == 3 and points:
            points.pop()
        redraw()

    def on_key(event):
        if event.key == "enter":
            if len(points) == count:
                plt.close(fig)
            else:
                print(f"Need {count} points before confirming.", flush=True)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()
    if len(points) != count:
        raise RuntimeError("selection cancelled before all points were picked")
    return np.array(points)


def computeH(im1_pts, im2_pts):

    im1_pts = np.asarray(im1_pts, dtype=float)
    im2_pts = np.asarray(im2_pts, dtype=float)
    if im1_pts.ndim != 2 or im1_pts.shape[1] != 2 or im1_pts.shape != im2_pts.shape or len(im1_pts) < 4:
        raise ValueError("need matching n x 2 point arrays, n >= 4")

    A = np.zeros((2 * len(im1_pts), 8))
    b = np.zeros(2 * len(im1_pts))
    for i, ((x, y), (u, v)) in enumerate(zip(im1_pts, im2_pts)):
        A[2 * i]     = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i], b[2 * i + 1] = u, v

    params, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 8:
        raise ValueError("points don't determine a unique homography")

    return np.array([[params[0], params[1], params[2]],
                     [params[3], params[4], params[5]],
                     [params[6], params[7], 1.0]])


def _output_bounds(im, H, output_bounds):

    h, w = im.shape[:2]
    if output_bounds is not None:
        return output_bounds
    corners = np.array([[0, 0, 1], [w - 1, 0, 1],
                        [w - 1, h - 1, 1], [0, h - 1, 1]])
    mapped = (H @ corners.T).T
    if mapped[:, 2].min() <= 0 <= mapped[:, 2].max():
        raise ValueError("image crosses the homography horizon")
    mapped = mapped[:, :2] / mapped[:, 2:3]
    x0, y0 = np.floor(mapped.min(axis=0)).astype(int)
    x1, y1 = np.ceil(mapped.max(axis=0)).astype(int)
    return (x0, y0, x1, y1)


def _backproject(im, H, bounds):

    h, w = im.shape[:2]
    x0, y0, x1, y1 = bounds
    u, v = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    dst = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    src = np.einsum("ij,jk->ik", np.linalg.inv(H), dst)
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = src[0] / src[2], src[1] / src[2]
    ok = (np.isfinite(x) & np.isfinite(y)
          & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1))
    return x, y, ok, u.shape


def warpImageNearestNeighbor(im, H, output_bounds=None):
    bounds = _output_bounds(im, H, output_bounds)
    x, y, ok, shape = _backproject(im, H, bounds)
    out = np.zeros(shape + im.shape[2:], dtype=im.dtype)
    out.reshape((ok.size,) + im.shape[2:])[ok] = \
        im[np.rint(y[ok]).astype(int), np.rint(x[ok]).astype(int)]
    return out, ok.reshape(shape), (bounds[0], bounds[1])


def warpImageBilinear(im, H, output_bounds=None):
    bounds = _output_bounds(im, H, output_bounds)
    x, y, ok, shape = _backproject(im, H, bounds)
    h, w = im.shape[:2]

    xv, yv = x[ok], y[ok]
    xa, ya = np.floor(xv).astype(int), np.floor(yv).astype(int)
    xb, yb = np.minimum(xa + 1, w - 1), np.minimum(ya + 1, h - 1)
    dx, dy = xv - xa, yv - ya
    if im.ndim == 3:
        dx, dy = dx[:, None], dy[:, None]

    colors = ((1 - dx) * (1 - dy) * im[ya, xa]
              + dx * (1 - dy) * im[ya, xb]
              + (1 - dx) * dy * im[yb, xa]
              + dx * dy * im[yb, xb])

    out = np.zeros(shape + im.shape[2:], dtype=float)
    out.reshape((ok.size,) + im.shape[2:])[ok] = colors
    if np.issubdtype(im.dtype, np.integer):
        lim = np.iinfo(im.dtype)
        out = np.clip(np.rint(out), lim.min, lim.max)
    return out.astype(im.dtype), ok.reshape(shape), (bounds[0], bounds[1])


def feather_weights(im):

    h, w = im.shape[:2]
    y, x = np.indices((h, w))
    return np.minimum.reduce([x + 1, w - x, y + 1, h - y]).astype(float)


def _place_on_canvas(canvas_shape, contributions):

    color_sum = np.zeros(canvas_shape + (3,), dtype=float)
    weight_sum = np.zeros(canvas_shape, dtype=float)
    for im, wt, (ox, oy) in contributions:
        h, w = im.shape[:2]
        sl = np.s_[oy:oy + h, ox:ox + w]
        color_sum[sl] += im.astype(float) * wt[:, :, None]
        weight_sum[sl] += wt
    covered = weight_sum > 0
    out = np.zeros_like(color_sum)
    out[covered] = color_sum[covered] / weight_sum[covered, None]
    return out, covered, weight_sum


def make_mosaic(im1, im2, H):

    warped, mask, (wx, wy) = warpImageBilinear(im1, H)
    ww, _, _ = warpImageBilinear(feather_weights(im1), H)
    ww *= mask
    ref_w = feather_weights(im2)

    x0 = min(wx, 0)
    y0 = min(wy, 0)
    x1 = max(wx + warped.shape[1] - 1, im2.shape[1] - 1)
    y1 = max(wy + warped.shape[0] - 1, im2.shape[0] - 1)
    shape = (y1 - y0 + 1, x1 - x0 + 1)

    out, covered, _ = _place_on_canvas(shape, [
        (warped, ww, (wx - x0, wy - y0)),
        (im2, ref_w, (0 - x0, 0 - y0)),
    ])
    if np.issubdtype(im1.dtype, np.integer):
        lim = np.iinfo(im1.dtype)
        out = np.clip(np.rint(out), lim.min, lim.max)
    return out.astype(im1.dtype), covered, (x0, y0)


def blend_images(images, transforms):

    layers = []
    for im, H in zip(images, transforms):
        warped, mask, origin = warpImageBilinear(im, H)
        w, _, _ = warpImageBilinear(feather_weights(im), H)
        layers.append((warped, w * mask, origin))

    x0 = min(o[0] for _, _, o in layers)
    y0 = min(o[1] for _, _, o in layers)
    x1 = max(o[0] + im.shape[1] - 1 for im, _, o in layers)
    y1 = max(o[1] + im.shape[0] - 1 for im, _, o in layers)
    shape = (y1 - y0 + 1, x1 - x0 + 1)

    out, covered, _ = _place_on_canvas(shape, [
        (im, wt, (ox - x0, oy - y0)) for im, wt, (ox, oy) in layers
    ])
    dtype = images[0].dtype
    if np.issubdtype(dtype, np.integer):
        lim = np.iinfo(dtype)
        out = np.clip(np.rint(out), lim.min, lim.max)
    return out.astype(dtype), covered, (x0, y0)


def show_pair_alignment(reference, moving, reference_points, moving_points, title):

    H_forward = computeH(reference_points, moving_points)
    warped, mask, (x, y) = warpImageBilinear(moving, np.linalg.inv(H_forward))
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    for ax, image, points, name in zip(
            axes[:2], [reference, moving], [reference_points, moving_points],
            ["Left image", "Right image"]):
        ax.imshow(image)
        ax.scatter(points[:, 0], points[:, 1], c="yellow", marker="+")
        for n, (px, py) in enumerate(points, 1):
            ax.annotate(str(n), (px, py), color="yellow",
                        xytext=(5, 5), textcoords="offset points")
        ax.set_title(name + ": selected features")
    axes[2].imshow(reference)
    h, w = warped.shape[:2]
    axes[2].imshow(warped, alpha=0.5 * mask,
                   extent=(x - 0.5, x + w - 0.5, y + h - 0.5, y - 0.5))
    axes[2].set_xlim(-0.5, reference.shape[1] - 0.5)
    axes[2].set_ylim(reference.shape[0] - 0.5, -0.5)
    axes[2].set_title("Overlap: inspect doubled edges")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    code_dir = __file__.rsplit("/", 1)[0]
    im1 = plt.imread(f"{code_dir}/../images/bank_01.jpg")
    im2 = plt.imread(f"{code_dir}/../images/bank_03.jpg")
    im3 = plt.imread(f"{code_dir}/../images/bank_04.jpg")
    im4 = plt.imread(f"{code_dir}/../images/bank_02.jpg")

    with np.load(f"{code_dir}/bank_01_03_points.npz") as data:
        pts1, pts2 = data["pts1"], data["pts2"]


    pair_03_04 = f"{code_dir}/bank_03_04_points.npz"
    try:
        with np.load(pair_03_04) as data:
            pts03, pts04 = data["pts1"], data["pts2"]
    except FileNotFoundError:
        print("Pick 8 shared features in Bank 03, then Bank 04 in the same order.")
        pts03 = pick_points(im2, "Bank 03 -> Bank 04")
        pts04 = pick_points(im3, "Bank 04: same features, same order")
        np.savez(pair_03_04, pts1=pts03, pts2=pts04)

    show_pair_alignment(im2, im3, pts03, pts04,
                        "Bank 03/04 alignment - close to continue")
    plt.show()

    if input("Redo 03/04 with 12 points? [y/N]: ").strip().lower() == "y":
        new03 = pick_points(im2, "Bank 03: 12 shared features", count=12)
        new04 = pick_points(im3, "Bank 04: same 12, same order", count=12)
        computeH(new03, new04)
        show_pair_alignment(im2, im3, new03, new04, "Proposed 12-point matches")
        plt.show()
        if input("Save? [y/N]: ").strip().lower() == "y":
            np.savez(f"{code_dir}/bank_03_04_points_previous.npz",
                     pts1=pts03, pts2=pts04)
            np.savez(pair_03_04, pts1=new03, pts2=new04)
            pts03, pts04 = new03, new04

    H_03_to_04 = computeH(pts03, pts04)
    H_04_to_03 = np.linalg.inv(H_03_to_04)


    pair_04_02 = f"{code_dir}/bank_04_02_points.npz"
    try:
        with np.load(pair_04_02) as data:
            pts04_f, pts02 = data["pts1"], data["pts2"]
    except FileNotFoundError:
        print("Pick 12 features visible in BOTH Bank 04 and Bank 02.")
        pts04_f = pick_points(im3, "Bank 04: 12 shared", count=12)
        pts02 = pick_points(im4, "Bank 02: same 12", count=12)
        computeH(pts04_f, pts02)
        np.savez(pair_04_02, pts1=pts04_f, pts2=pts02)

    H_04_to_02 = computeH(pts04_f, pts02)
    H_02_to_03 = H_04_to_03 @ np.linalg.inv(H_04_to_02)


    for label, H, src, dst in [
            ("03->04", H_03_to_04, pts03, pts04),
            ("04->02", H_04_to_02, pts04_f, pts02)]:
        proj = (H @ np.column_stack([src, np.ones(len(src))]).T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err = np.linalg.norm(proj - dst, axis=1)
        print(f"Bank {label} mean error: {err.mean():.2f}px")

    H = computeH(pts1, pts2)
    proj = (H @ np.column_stack([pts1, np.ones(len(pts1))]).T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - pts2, axis=1)
    print(f"Bank 01->03 mean error: {err.mean():.2f}px")

    mosaic, _, _ = make_mosaic(im1, im2, H)
    mosaic_three, _, _ = blend_images([im1, im2, im3], [H, np.eye(3), H_04_to_03])
    mosaic_four, _, _ = blend_images([im1, im2, im3, im4],
                                     [H, np.eye(3), H_04_to_03, H_02_to_03])

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, image, title in zip(axes, [mosaic, mosaic_three, mosaic_four],
                                ["Two photos", "Three photos", "Four photos"]):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.show()
