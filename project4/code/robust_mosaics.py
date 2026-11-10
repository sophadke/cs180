from pathlib import Path
import json
import itertools
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from harris import get_harris_corners
from corners import anms
from features import extract_descriptors, match_features
from main import computeH, blend_images


def apply_H(H, pts):

    ph = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    ok = np.abs(ph[:, 2]) > 1e-10
    out = np.full((len(pts), 2), np.inf)
    out[ok] = ph[ok, :2] / ph[ok, 2, None]
    return out


project = apply_H

def normalized_dlt(src, dst):

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("Need at least four corresponding 2D points")

    def norm(pts):
        c = pts.mean(axis=0)
        distance = np.linalg.norm(pts - c, axis=1).mean()
        if distance < 1e-10:
            raise ValueError("Coincident points")
        s = np.sqrt(2) / distance
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
        return apply_H(T, pts), T

    p, Tp = norm(src)
    q, Tq = norm(dst)

    A = []
    for (x, y), (u, v) in zip(p, q):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A)
    if np.linalg.matrix_rank(A) < 8:
        raise ValueError("degenerate correspondences")

    _, _, Vt = np.linalg.svd(A, full_matrices=len(A) < 9)
    H = np.linalg.inv(Tq) @ Vt[-1].reshape(3, 3) @ Tp
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-10 or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Singular homography")
    return H / H[2, 2]


def nondegenerate(pts):

    span = max(float(np.ptp(pts, axis=0).max()), 1.0)
    for a, b, c in itertools.combinations(pts, 3):
        u, v = b - a, c - a
        if abs(u[0] * v[1] - u[1] * v[0]) < 1e-4 * span * span:
            return False
    return True


def ransac_homography(src, dst, threshold=3.0, max_trials=10000,
                       confidence=0.999, seed=180):

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("At least four matches required")
    rng = np.random.default_rng(seed)

    best, best_key = None, (-1, -np.inf)
    budget, trials = max_trials, 0
    while trials < budget:
        trials += 1
        ids = rng.choice(len(src), 4, replace=False)
        if not nondegenerate(src[ids]) or not nondegenerate(dst[ids]):
            continue
        try:
            H = normalized_dlt(src[ids], dst[ids])
        except (ValueError, np.linalg.LinAlgError):
            continue
        err = np.linalg.norm(apply_H(H, src) - dst, axis=1)
        mask = err < threshold
        n = int(mask.sum())
        key = (n, -np.median(err[mask]) if n else -np.inf)
        if key > best_key:
            best, best_key = (H, mask), key
            w = n / len(src)
            if w >= 1:
                budget = trials
            elif w > 0:
                need = int(np.ceil(np.log1p(-confidence) / np.log1p(-w ** 4)))
                budget = min(budget, max(trials, need))

    if best is None or best[1].sum() < 4:
        raise ValueError("no valid RANSAC consensus")

    H, mask = best
    for _ in range(10):
        try:
            H2 = normalized_dlt(src[mask], dst[mask])
        except (ValueError, np.linalg.LinAlgError):
            break
        err = np.linalg.norm(apply_H(H2, src) - dst, axis=1)
        new_mask = err < threshold
        if new_mask.sum() < 4:
            break
        H = H2
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    H = normalized_dlt(src[mask], dst[mask])
    err = np.linalg.norm(apply_H(H, src) - dst, axis=1)
    if (err < threshold).sum() < 4:
        raise ValueError("Insufficient refined inliers")
    return H, err < threshold, err, trials


def multiscale_features(image, scales=(1.0, 2 ** -0.5, 0.5), count=500):

    gray = (image.astype(float) / 255
            * np.array([0.2125, 0.7154, 0.0721])).sum(axis=2)
    pts, desc, levels, strengths, per_level = [], [], [], [], []
    for lvl, s in enumerate(scales):
        shape = tuple(np.round(np.array(gray.shape) * s).astype(int))
        small = gray if s == 1 else resize(gray, shape, anti_aliasing=True,
                                           preserve_range=True)
        response, yx = get_harris_corners(small, edge_discard=20)
        coords = yx.T[:, ::-1]
        vals = response[yx[0], yx[1]]
        sel, _ = anms(coords, vals, count)
        valid, d, kept = extract_descriptors(small, coords[sel])

        factor = np.array(gray.shape[::-1]) / np.array(small.shape[::-1])
        pts.append((valid + 0.5) * factor - 0.5)
        desc.append(d)
        levels += [lvl] * len(d)
        strengths += list(vals[sel][kept])
        per_level.append(dict(scale=s, shape=list(small.shape),
                              candidates=len(coords), descriptors=len(d)))
    return (np.concatenate(pts), np.concatenate(desc),
            np.array(levels), np.array(strengths), per_level)


def unique_matches(d, e):

    matches, ratios, _ = match_features(d, e, 0.8)
    order = np.argsort(ratios, kind="stable")
    seen, keep = set(), []
    for i in order:
        j = int(matches[i, 1])
        if j not in seen:
            keep.append(i)
            seen.add(j)
    return matches[keep], ratios[keep], len(matches)


def check_canvas(images, transforms, max_pixels=16_000_000):

    corners = []
    for im, H in zip(images, transforms):
        h, w = im.shape[:2]
        p = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        w_denom = np.column_stack([p, np.ones(4)]) @ H[2]
        if w_denom.min() <= 0 <= w_denom.max():
            raise ValueError("horizon crosses image")
        corners.append(apply_H(H, p))
    corners = np.concatenate(corners)
    size = np.ceil(corners.max(0)) - np.floor(corners.min(0)) + 1
    if not np.isfinite(size).all() or size.prod() > max_pixels:
        raise ValueError("implausibly large mosaic canvas")


def diagnostic(images, p, q, matches, mask, path, label):
    left, right = images
    canvas = np.zeros((max(left.shape[0], right.shape[0]),
                       left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1]:] = right
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(canvas)
    for keep, color in [(False, "#ff4848"), (True, "#38f078")]:
        for (i, j) in matches[mask == keep]:
            a = p[i]
            b = q[j] + [left.shape[1], 0]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.45, alpha=0.6)
    ax.set_title(f"{label}: {mask.sum()}/{len(matches)} inliers "
                 "(green), rejected (red)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    scenes = [
        ("bank",   "bank_01",  "bank_03", "bank_01_03_points.npz"),
        ("street", "street_04", "street_02", "street_04_02_points.npz"),
        ("river",  "river_01", "river_02", "river_01_02_points.npz"),
    ]
    scales = (1.0, 2 ** -0.5, 0.5)
    metrics = {"parameters": dict(scales=list(scales), corners_per_scale=500, ratio=0.8, ransac_threshold_px=3, max_trials=10000, confidence=0.999, seed=180, max_image_side=1024), "scenes": {}}

    for label, a, b, manual_file in scenes:
        originals = [imread(root / "images" / f"{n}.jpg") for n in (a, b)]
        images, factors = [], []
        for im in originals:
            s = min(1.0, 1024 / max(im.shape[:2]))
            shape = tuple(np.round(np.array(im.shape[:2]) * s).astype(int))
            small = im if s == 1 else np.rint(
                resize(im, shape, anti_aliasing=True,
                       preserve_range=True)).astype(np.uint8)
            images.append(small)
            factors.append(np.array(small.shape[1::-1])
                           / np.array(im.shape[1::-1]))

        feats = []
        for name, im in zip((a, b), images):
            p, d, lv, st, per = multiscale_features(im, scales)
            np.savez(root / "code" / f"{name}_multiscale.npz",
                     coords=p, descriptors=d, levels=lv, strengths=st)
            feats.append((p, d, lv, st, per))

        (p, d, lp, _, _), (q, e, lq, _, _) = feats
        matches, ratios, before_unique = unique_matches(d, e)
        H, mask, err, trials = ransac_homography(
            p[matches[:, 0]], q[matches[:, 1]])

        with np.load(root / "code" / manual_file) as pts:
            manual_H = computeH(
                (pts["pts1"] + 0.5) * factors[0] - 0.5,
                (pts["pts2"] + 0.5) * factors[1] - 0.5)

        mosaics = {}
        for mode, T in (("manual", manual_H), ("auto", H)):
            check_canvas(images, [T, np.eye(3)])
            mosaic, _, _ = blend_images(images, [T, np.eye(3)])
            imsave(root / "results" / f"{label}_{mode}_b4.png",
                   mosaic, check_contrast=False)
            mosaics[mode] = mosaic

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for ax, key, title in zip(
                axes, ("manual", "auto"),
                ("Manual correspondences", "Multiscale + 4-point RANSAC")):
            ax.imshow(mosaics[key])
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{label.title()}: same warp, same blend")
        fig.tight_layout()
        fig.savefig(root / "results" / f"{label}_manual_auto.png", dpi=150)
        plt.close(fig)

        diagnostic(images, p, q, matches, mask,
                   root / "results" / f"{label}_ransac.png", label.title())

        single_matches, _, _ = unique_matches(d[lp == 0], e[lq == 0])
        single_stats = dict(matches=len(single_matches))
        try:
            _, sm, se, _ = ransac_homography(p[lp == 0][single_matches[:, 0]], q[lq == 0][single_matches[:, 1]])
            single_stats.update(inliers=int(sm.sum()), median_error_px=float(np.median(se[sm])))
        except ValueError as ex:
            single_stats.update(inliers=0, error=str(ex))
        cross = lp[matches[mask, 0]] != lq[matches[mask, 1]]
        np.savez(root / "code" / f"{label}_ransac.npz",
                 H=H, manual_H=manual_H, matches=matches, ratios=ratios,
                 inliers=mask, errors=err,
                 points1=p[matches[:, 0]], points2=q[matches[:, 1]])

        metrics["scenes"][label] = {
            "images": [a, b],
            "working_shapes": [list(im.shape) for im in images],
            "levels": [feats[0][4], feats[1][4]],
            "ratio_passes": before_unique,
            "single_scale": single_stats,
            "mean_error_px": float(err[mask].mean()),
            "matches": len(matches),
            "inliers": int(mask.sum()),
            "median_error_px": float(np.median(err[mask])),
            "trials": trials,
            "cross_scale_inliers": int(cross.sum()),
            "H": H.tolist(),
        }
        print(label, json.dumps(metrics["scenes"][label]))


    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    image = imread(root / "images" / "bank_01.jpg")
    with np.load(root / "code" / "bank_01_multiscale.npz") as f:
        for lvl, (ax, s) in enumerate(zip(axes, scales)):
            ax.imshow(image)
            ids = np.flatnonzero(f["levels"] == lvl)
            ids = ids[np.argsort(-f["strengths"][ids])[:35]]
            for x, y in f["coords"][ids]:
                ax.add_patch(Circle((x, y), 20 / s, fill=False,
                                    color="#ff3030", lw=0.7))
            ax.set_title(f"scale {s:.3f}, support ~{40 / s:.0f}px")
            ax.axis("off")
    fig.suptitle("35 strongest features per level, bank_01", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(root / "results" / "multiscale_supports.png", dpi=150)
    plt.close(fig)

    (root / "results" / "ransac_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
