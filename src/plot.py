import numpy as np
import matplotlib.pyplot as plt

from src.controller import compress

arr_quant8 = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 63]
arr_zeroes8 = [1, 19, 38, 57, 76, 95, 114, 133, 152, 171, 190, 209, 228, 255]

arr_quant16 = [1, 19, 38, 57, 76, 95, 114, 133, 152, 171, 190, 209, 228, 255]
arr_zeroes16 = [1, 19, 38, 57, 76, 95, 114, 133, 152, 171, 190, 209, 228, 255]

arr_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

def plot_regression(x_data, y_data, color, label):
    x_data = np.clip(x_data, a_min=1e-10, a_max=None)
    log_x_data = np.log(x_data)
    coefficients = np.polyfit(log_x_data, y_data, 1)
    poly = np.poly1d(coefficients)

    x_values = np.linspace(np.min(x_data), np.max(x_data), 100)
    log_x_values = np.log(x_values)
    y_values = poly(log_x_values)

    plt.scatter(x_data, y_data, color=color, label=f"{label} - Data")
    plt.plot(x_values, y_values, label=f"{label} - Regression", color=color)

def get_y_data(type, photo_size, fast_mode):
    y_data8_psnr, y_data8_ssim = zip(*[compress(8, q, z, photo_size, fast_mode) for q, z in zip(arr_quant8, arr_zeroes8)])
    y_data16_psnr, y_data16_ssim = zip(*[compress(16, q, z, photo_size, fast_mode) for q, z in zip(arr_quant16, arr_zeroes16)])

    if type == "psnr":
        y_data1, y_data2 = y_data8_psnr, y_data16_psnr
    else:
        y_data1, y_data2 = y_data8_ssim, y_data16_ssim

    return np.array(y_data1), np.array(y_data2)

def draw_plot(type, photo_size, fast_mode):
    x_data1 = np.array(arr_x)

    y_data1, y_data2 = get_y_data(type, photo_size, fast_mode)

    plot_regression(x_data1, y_data1, "RED", "Блок 8х8")
    plot_regression(x_data1, y_data2, "BLUE", "Блок 16х16")

    plt.xlabel('n')
    plt.title(f'Изменение качества по коэфф-ту {type} \nв зависимости от коэфф-та квантования и зануления')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"image/plot/versus_{type}.png")
    plt.show()