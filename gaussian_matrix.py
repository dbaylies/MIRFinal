import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage


def gaussian_matrix(fs_chromagram, L):

    # Gaussian kernel lengths
    s_l_sec = 0.3 # seconds
    s_t_sec = 2 # Seconds
    variance = 0.16

    # Create bi-variate rectangular Gaussian kernel

    s_l = round(s_l_sec * fs_chromagram)
    s_t = round(s_t_sec * fs_chromagram)

    std_dev = np.sqrt(variance)

    g_l = np.matrix(signal.gaussian(s_l,std_dev))
    g_t = np.matrix(signal.gaussian(s_t,std_dev))

    G = np.matmul(np.transpose(g_l),g_t)

    # Convolve L with Gaussian kernel
    P = ndimage.convolve(L, G, mode='constant')

    return P
