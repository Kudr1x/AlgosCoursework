from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from src.image_processing import apply_dct_to_image, zero_blocks, quantize_blocks, apply_idct_to_image
from src.utils import pad_image


def compress(block_size, num_zeros, quant, size, fast_mode):
    image_path = f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/input.png'
    image = pad_image(Image.open(image_path), block_size)
    image_np = np.array(image)

    # Apply DCT
    dct_image = apply_dct_to_image(image_np, block_size, fast_mode)

    zeroed_image_pil = Image.fromarray(np.uint8(dct_image))
    zeroed_image_pil.save(f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/dct_output.png')

    # Zero blocks
    zeroed_dct_image = zero_blocks(dct_image.copy(), block_size, num_zeros)

    # Save zeroed image
    zeroed_image_pil = Image.fromarray(np.uint8(zeroed_dct_image))
    zeroed_image_pil.save(f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/zeroed_output.png')

    # Quantize blocks
    quantized_dct_image = quantize_blocks(zeroed_dct_image, block_size, quant)

    # Save quantized image
    quantized_image_pil = Image.fromarray(np.uint8(quantized_dct_image))
    quantized_image_pil.save(f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/quantized_output.png')

    # Apply IDCT
    reconstructed_image = apply_idct_to_image(quantized_dct_image, block_size, fast_mode)

    # Save reconstructed image
    reconstructed_image_pil = Image.fromarray(np.uint8(reconstructed_image))
    reconstructed_image_pil.save(f'/home/kudrix/PycharmProjects/AlgosCoursework/image/{size}/reconstructed_output.png')

    # Calculate PSNR and SSIM
    original_image_np = np.array(image)
    reconstructed_image_np = np.array(reconstructed_image_pil)

    # Calculate PSNR
    data_range = original_image_np.max() - original_image_np.min()
    psnr_value = psnr(original_image_np, reconstructed_image_np, data_range=data_range)
    print(f'PSNR: {psnr_value} dB')

    # Calculate SSIM
    ssim_value = ssim(original_image_np, reconstructed_image_np, channel_axis=-1, data_range=data_range)
    print(f'SSIM: {ssim_value}')