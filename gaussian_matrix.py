import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import matplotlib.pyplot as pyp


def gaussian_matrix(fs_chromagram, L, plot):

    # Gaussian kernel lengths
    s_l_sec = 0.5 # seconds
    s_t_sec = 30 # Seconds
    variance = 0.16

    # Create bi-variate rectangular Gaussian kernel

    s_l = round(s_l_sec * fs_chromagram)
    s_t = round(s_t_sec * fs_chromagram)

    std_dev = np.sqrt(variance)

    std_dev_l = std_dev * s_l
    std_dev_t = std_dev * s_t

    g_l = np.matrix(signal.gaussian(s_l,std_dev_l))
    g_t = np.matrix(signal.gaussian(s_t,std_dev_t))

    G = np.matmul(np.transpose(g_l),g_t)

    if plot:
        pyp.figure()
        pyp.plot(signal.gaussian(s_t,std_dev_t))
        pyp.title('g_t')

    if plot:
        pyp.figure()
        pyp.imshow(G, aspect='auto')
        pyp.title('Gaussian Kernel')

    # Convolve L with Gaussian kernel
    P = ndimage.convolve(L, G, mode='constant')

    if plot:
        pyp.figure()
        pyp.imshow(P)
        pyp.title('Smoothed Time-Lag Matrix')

    return P
