from PIL import Image
import numpy as np

def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

def save_image(image_np, output_path):
    image_pil = Image.fromarray(np.uint8(image_np))
    image_pil.save(output_path)

def pad_image(image, block_size):
    width, height = image.size
    new_width = (width + block_size - 1) // block_size * block_size
    new_height = (height + block_size - 1) // block_size * block_size
    new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    new_image.paste(image, (0, 0))
    return new_image

