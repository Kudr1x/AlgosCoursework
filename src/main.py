from src.plot import draw_plot

PHOTO_SIZE = "medium"
FAST_MODE  = True
TYPES = ["psnr", "ssim"]

if __name__ == "__main__":
    for t in TYPES:
        draw_plot(t, PHOTO_SIZE, FAST_MODE)