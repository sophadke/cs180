"""Part 1.1: convolution from scratch, numpy only.

These are the two functions from funfilter.py, moved here so other files
can import them without re-running the timing experiment.
"""
import numpy as np


def conv_four_loops(img, kernel):
    kernel = kernel[::-1, ::-1]                 # flip the kernel (convolution, not correlation)
    kh, kw = kernel.shape
    ph = kh // 2                                # zero padding above/below
    pw = kw // 2                                # zero padding left/right
    padded = np.pad(img, ((ph, ph), (pw, pw)))

    H, W = img.shape
    out = np.zeros((H, W))                      # "same" size output

    for i in range(H):                          # every output pixel
        for j in range(W):
            total = 0
            for u in range(kh):                 # every kernel entry
                for v in range(kw):
                    total += padded[i + u, j + v] * kernel[u, v]
            out[i, j] = total
    return out


def conv_two_loops(img, kernel):
    kernel = kernel[::-1, ::-1]
    kh, kw = kernel.shape
    padded = np.pad(img, ((kh // 2, (kh - 1) // 2), (kw // 2, (kw - 1) // 2)))

    H, W = img.shape
    out = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            patch = padded[i:i + kh, j:j + kw]  # the whole neighborhood at once
            out[i, j] = np.sum(patch * kernel)  # element-wise multiply, then add
    return out
