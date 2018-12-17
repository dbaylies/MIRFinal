import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as pyp


def get_chromagram(filepath):

    # Read in file
    x_t, fs = sf.read(filepath)

    # Generate stft
    N_fft = 16384
    bin_freqs, sample_times, Zxx = signal.stft(x_t, fs, window=np.hanning(N_fft), nperseg=N_fft)  # Overlap defaults to nperseg/2

    # Generate center frequencies
    bins_per_octave = 12
    num_octaves = 8
    f_min = 55 # Hz - A1
    bin_indices = np.arange(bins_per_octave * num_octaves)
    bin_center_freqs = f_min * (2 ** (bin_indices / bins_per_octave))

    # Generate filterbank matrix
    filterbank_matrix = np.zeros((len(bin_center_freqs),len(bin_freqs)))
    nearest_bin_idx = np.zeros(len(bin_center_freqs))

    for i in np.arange(len(bin_center_freqs)):
        nearest_bin_idx[i] = np.abs(bin_freqs - bin_center_freqs[i]).argmin()
        # Get length of window

    for i in np.arange(len(bin_center_freqs)):
        if i == 0:
            win_start = nearest_bin_idx[i] - (nearest_bin_idx[i+1] - nearest_bin_idx[i])
            win_end = nearest_bin_idx[i+1]
        elif i == len(bin_center_freqs)-1:
            win_start = nearest_bin_idx[i-1]
            win_end = nearest_bin_idx[i] + (nearest_bin_idx[i] - nearest_bin_idx[i-1])
        else:
            win_start = nearest_bin_idx[i-1]
            win_end = nearest_bin_idx[i+1]

        # Insert window into filterbank matrix
        win_start = int(win_start)
        win_end = int(win_end)
        win_length = win_end-win_start+1
        win = np.blackman(win_length)
        win_norm = win/np.sum(win)
        filterbank_matrix[i,win_start:win_end+1] = win_norm

    # Apply filterbank to get log-frequency spectrum

    log_freq_spectrum = np.matmul(filterbank_matrix,Zxx)

    # Sum across octaves
    chromagram = np.zeros((bins_per_octave,len(sample_times)))

    for i in np.arange(bins_per_octave):
        for j in np.arange(num_octaves):
            chromagram[i,:] += abs(log_freq_spectrum[j*bins_per_octave+i,:])

    fs_chromagram = fs/(N_fft/2)

    pyp.figure()
    pyp.imshow(abs(chromagram), aspect='auto', origin='lower',extent=[sample_times[0],sample_times[-1],0,bins_per_octave])
    pyp.yticks(np.arange(12)+0.5, ['A','A#','B','C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'])
    pyp.title('Chromagram for ' + filepath)

    return chromagram, fs_chromagram
