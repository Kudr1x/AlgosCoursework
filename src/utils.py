from PIL import Image
import numpy as np

def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

def save_image(image_np, output_path):
    image_pil = Image.fromarray(np.uint8(image_np))
    image_pil.save(output_path)