"""Convolution functions for Part 1.1."""

import numpy as np


def conv_four_loops(img, kernel):
    flipped_kernel = kernel[::-1, ::-1]
    kernel_height, kernel_width = flipped_kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)))

    height, width = img.shape
    output = np.zeros((height, width))

    for row in range(height):
        for col in range(width):
            total = 0
            for kernel_row in range(kernel_height):
                for kernel_col in range(kernel_width):
                    pixel = padded_img[row + kernel_row, col + kernel_col]
                    weight = flipped_kernel[kernel_row, kernel_col]
                    total += pixel * weight
            output[row, col] = total

    return output


def conv_two_loops(img, kernel):
    flipped_kernel = kernel[::-1, ::-1]
    kernel_height, kernel_width = flipped_kernel.shape

    top = kernel_height // 2
    bottom = (kernel_height - 1) // 2
    left = kernel_width // 2
    right = (kernel_width - 1) // 2
    padded_img = np.pad(img, ((top, bottom), (left, right)))

    height, width = img.shape
    output = np.zeros((height, width))

    for row in range(height):
        for col in range(width):
            patch = padded_img[
                row : row + kernel_height,
                col : col + kernel_width,
            ]
            output[row, col] = np.sum(patch * flipped_kernel)

    return output
