import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as pyp

def peak_pick(c, fs):


    # w_c = 4/((fs)/2)
    medfilt_len = 17
    offset = 0.03

    # design 1st order butterworth
    b, a = signal.butter(1, 0.2)

    # apply filter
    n_t_smoothed = signal.filtfilt(b, a, c)

    # normalize
    max_val = np.amax(n_t_smoothed)
    n_t_norm = n_t_smoothed/max_val

    # Apply median filer
    n_t_median = signal.medfilt(n_t_norm, medfilt_len)

    # Apply offset
    thresh = n_t_median + offset

    # Peak picking
    ndarray, dict = signal.find_peaks(n_t_norm, thresh)

    # Get onset amplitudes
    onset_a = c[[ndarray]]

    # Get times of detected onsets
    onset_t = ndarray * (1/fs)

    pyp.plot(n_t_norm)
    pyp.plot(ndarray,onset_a,'ro')
    pyp.title('Smoothed novelty plot and found peaks')

    return onset_a, onset_t