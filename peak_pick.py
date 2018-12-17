import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as pyp

def peak_pick(c, fs_chromagram, look_back, seg_times, plot):

    # parameters
    w_c = 0.05
    medfilt_len = 33
    offset = 0.01

    # design 1st order butterworth
    b, a = signal.butter(1, w_c)

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
    onset_t = ndarray * (1 / fs_chromagram)

    # Add time offset from lag matrix step
    onset_t = onset_t + look_back/2

    seg_times = np.array(seg_times)

    # Remove lookback offset
    seg_times = seg_times - look_back/2
    seg_times = seg_times * fs_chromagram

    if plot:
        pyp.plot(n_t_norm)
        pyp.plot(ndarray, onset_a, 'ro')
        for xc in seg_times:
            pyp.axvline(x=xc)
        pyp.title('Smoothed novelty plot and found peaks')

    return onset_a, onset_t
