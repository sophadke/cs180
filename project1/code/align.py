import numpy as np
from scipy.ndimage import gaussian_filter, sobel as ndi_sobel


def feat_identity(img):
    return img


def feat_percentile_stretch(img, lo=1, hi=99):
    lo_v, hi_v = np.percentile(img, [lo, hi])
    if hi_v <= lo_v:
        return img
    out = (img - lo_v) / (hi_v - lo_v)
    return np.clip(out, 0, 1)


def feat_hist_eq(img, bins=256):
    flat = img.ravel()
    hist, bin_edges = np.histogram(flat, bins=bins, range=(0, 1))
    cdf = hist.cumsum().astype(np.float64)
    cdf /= cdf[-1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    equalized = np.interp(flat, bin_centers, cdf)
    return equalized.reshape(img.shape)


def feat_sobel(img):

    gx = ndi_sobel(img, axis=1)
    gy = ndi_sobel(img, axis=0)
    return np.hypot(gx, gy)


FEATURES = {
    "identity": feat_identity,
    "percentile": feat_percentile_stretch,
    "histeq": feat_hist_eq,
    "sobel": feat_sobel,
}

def metric_l2(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def metric_ncc(a, b):
    # Return negative NCC so that "smaller is better" stays consistent
    # with the other metrics.
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    ncc = np.dot(a.ravel(), b.ravel()) / denom
    return float(-ncc)


METRICS = {
    "l2": metric_l2,
    "ncc": metric_ncc,
}


def crop_border(img, frac):

    if frac <= 0:
        return img
    h, w = img.shape[:2]
    dy, dx = int(h * frac), int(w * frac)
    return img[dy:h - dy, dx:w - dx]


def shift(img, dy, dx):

    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def downsample2(img):

    blurred = gaussian_filter(img, sigma=1.0)
    return blurred[::2, ::2]


# ----------------------------------------------------------------------
# Alignment search
# ----------------------------------------------------------------------

def align_single_scale(moving, ref, window=15, metric="ncc", feature="identity",
                       border_frac=0.1):
    # Brute-force search over dy, dx in [-window, window].
    metric_fn = METRICS[metric]
    feature_fn = FEATURES[feature]

    ref_c = crop_border(ref, border_frac)
    ref_feat = feature_fn(ref_c)

    best_score = None
    best_dy, best_dx = 0, 0

    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            candidate = shift(moving, dy, dx)
            candidate_c = crop_border(candidate, border_frac)
            candidate_feat = feature_fn(candidate_c)
            s = metric_fn(candidate_feat, ref_feat)
            if best_score is None or s < best_score:
                best_score = s
                best_dy, best_dx = dy, dx

    return best_dy, best_dx, best_score


def align_pyramid(moving, ref, window=15, metric="ncc", feature="identity", border_frac=0.1, min_size=400, refine_window=4):

    h, w = ref.shape[:2]

    if max(h, w) <= min_size:
        return align_single_scale(moving, ref, window=window, metric=metric,
                                  feature=feature, border_frac=border_frac)

    moving_small = downsample2(moving)
    ref_small = downsample2(ref)

    dy0, dx0, _ = align_pyramid(moving_small, ref_small, window=window,
                                metric=metric, feature=feature,
                                border_frac=border_frac, min_size=min_size,
                                refine_window=refine_window)
    dy0, dx0 = dy0 * 2, dx0 * 2

    pre_shifted = shift(moving, dy0, dx0)
    rdy, rdx, best_score = align_single_scale(
        pre_shifted, ref, window=refine_window, metric=metric,
        feature=feature, border_frac=border_frac,
    )

    return dy0 + rdy, dx0 + rdx, best_score



def split_channels(img):

    h = img.shape[0] // 3
    b = img[0:h]
    g = img[h:2 * h]
    r = img[2 * h:3 * h]
    return b, g, r


def colorize(img_float, use_pyramid=True, window=15, metric="ncc",
             feature="identity", border_frac=0.1, min_size=400,
             refine_window=4):

    b, g, r = split_channels(img_float)

    align_fn = align_pyramid if use_pyramid else align_single_scale
    kwargs = dict(window=window, metric=metric, feature=feature,
                  border_frac=border_frac)
    if use_pyramid:
        kwargs.update(min_size=min_size, refine_window=refine_window)

    dy_g, dx_g, score_g = align_fn(g, b, **kwargs)
    dy_r, dx_r, score_r = align_fn(r, b, **kwargs)

    g_aligned = shift(g, dy_g, dx_g)
    r_aligned = shift(r, dy_r, dx_r)

    rgb = np.dstack([r_aligned, g_aligned, b])
    rgb = np.clip(rgb, 0, 1)

    return rgb, (dy_g, dx_g, score_g), (dy_r, dx_r, score_r)