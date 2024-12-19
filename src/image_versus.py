from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def get_ssim(original_image_np, reconstructed_image_np):
    data_range = original_image_np.max() - original_image_np.min()
    return ssim(original_image_np, reconstructed_image_np, channel_axis=-1, data_range=data_range)

def get_psnr(original_image_np, reconstructed_image_np):
    data_range = original_image_np.max() - original_image_np.min()
    return psnr(original_image_np, reconstructed_image_np, data_range=data_range)