from PIL import Image
import numpy as np

from src.image_processing import apply_dct_to_image, zero_blocks, quantize_blocks, apply_idct_to_image
from src.image_versus import get_ssim, get_psnr
from src.utils import pad_image


def compress(block_size, num_zeros, quant, size, fast_mode):
    image_path = f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/input.png'
    image = pad_image(Image.open(image_path), block_size)
    image_np = np.array(image)

    dct_image = apply_dct_to_image(image_np, block_size, fast_mode)

    zeroed_image_pil = Image.fromarray(np.uint8(dct_image))
    zeroed_image_pil.save(f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/dct_output.png')

    zeroed_dct_image = zero_blocks(dct_image.copy(), block_size, num_zeros)

    zeroed_image_pil = Image.fromarray(np.uint8(zeroed_dct_image))
    zeroed_image_pil.save(f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/zeroed_output.png')

    quantized_dct_image = quantize_blocks(zeroed_dct_image, block_size, quant)

    quantized_image_pil = Image.fromarray(np.uint8(quantized_dct_image))
    quantized_image_pil.save(f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/quantized_output.png')

    reconstructed_image = apply_idct_to_image(quantized_dct_image, block_size, fast_mode)

    reconstructed_image_pil = Image.fromarray(np.uint8(reconstructed_image))
    reconstructed_image_pil.save(f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/reconstructed_output.png')

    original_image_np = np.array(image)
    reconstructed_image_np = np.array(reconstructed_image_pil)

    return float(get_psnr(original_image_np, reconstructed_image_np)), float(get_ssim(original_image_np, reconstructed_image_np))