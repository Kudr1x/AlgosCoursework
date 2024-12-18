import scipy.fftpack

def dct2(block):
    return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')