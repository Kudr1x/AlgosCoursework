import numpy as np
from dct import dct2, idct2

def apply_dct_to_image(image, block_size):
    h, w, _ = image.shape
    dct_image = np.zeros_like(image, dtype=float)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            for k in range(3):  # Apply DCT to each color channel
                block = image[i:i + block_size, j:j + block_size, k]
                if block.shape == (block_size, block_size):
                    dct_image[i:i + block_size, j:j + block_size, k] = dct2(block)

    return dct_image

def apply_idct_to_image(dct_image, block_size):
    h, w, _ = dct_image.shape
    image = np.zeros_like(dct_image, dtype=float)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            for k in range(3):  # Apply IDCT to each color channel
                block = dct_image[i:i + block_size, j:j + block_size, k]
                if block.shape == (block_size, block_size):
                    image[i:i + block_size, j:j + block_size, k] = idct2(block)

    return image

def zero_blocks(dct_image, block_size, num_zeros):
    h, w, _ = dct_image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            for k in range(3):  # Zero blocks in each color channel
                block = dct_image[i:i + block_size, j:j + block_size, k]
                if block.shape == (block_size, block_size):
                    indices = np.unravel_index(np.argsort(block, axis=None)[:num_zeros], block.shape)
                    block[indices] = 0
    return dct_image

def quantize_blocks(dct_image, block_size, quant):
    h, w, _ = dct_image.shape
    quantized_image = np.zeros_like(dct_image)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            for k in range(3):  # Quantize blocks in each color channel
                block = dct_image[i:i + block_size, j:j + block_size, k]
                if block.shape == (block_size, block_size):
                    quantized_image[i:i + block_size, j:j + block_size, k] = np.round(block / quant) * quant

    return quantized_image