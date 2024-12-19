import numpy as np
import scipy.fftpack

def dct2(block, fast_mode):
    if fast_mode:
        return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')
    return dct(dct(block.T).T)

def idct2(block, fast_mode):
    if fast_mode:
        return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')
    return idct(idct(block.T).T)

def dct(x):
    N = x.shape[0]
    X = np.zeros_like(x, dtype=np.float64)
    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += x[n] * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
        X[k] = sum_val
    return X

def idct(X):
    N = X.shape[0]
    x = np.zeros_like(X, dtype=np.float64)
    for n in range(N):
        sum_val = 0
        for k in range(N):
            sum_val += X[k] * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
        x[n] = sum_val / N
    return x