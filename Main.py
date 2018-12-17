import numpy as np
import matplotlib.pyplot as pyp
import get_chromagram, recurrence_matrix, gaussian_matrix, novelty, timelag_matrix, peak_pick, get_metrics, seg_parser
import os

# Flag for showing plots
plot = 1

audio_root = '../Audio/'

# Which album do you want to look at?
album_num = 0

albums = os.listdir(audio_root)
songs = os.listdir(audio_root + albums[album_num])

seg_times, files = seg_parser.seg_parse(album_num)

precision = 0
recall = 0
f_measure = 0

for i in np.arange(len(files)):

    filepath = audio_root + albums[album_num] + '/' + songs[i]

    chromagram, fs_chromagram = get_chromagram.get_chromagram(filepath, plot)
    N, recurrence, look_back = recurrence_matrix.recurrence_matrix(chromagram,fs_chromagram, plot)
    L = timelag_matrix.timelag_matrix(N, recurrence, plot)
    P = gaussian_matrix.gaussian_matrix(fs_chromagram, L, plot)
    c = novelty.novelty(P, plot)

    # Get relevant array
    seg_times_i = seg_times[files[i]]

    onset_a, onset_t = peak_pick.peak_pick(c, fs_chromagram, look_back, seg_times_i, plot)
    precision_, recall_, f_measure_ = get_metrics.get_metrics(seg_times_i, files, onset_t)

    precision += precision_
    recall += recall_
    f_measure += f_measure_

num_songs = len(files)

precision_avg = precision/num_songs
recall_avg = recall/num_songs
f_measure_avg = f_measure/num_songs

print('For ' + albums[album_num])
print('\nP: ' + str(precision_avg) + '\nR: ' + str(recall_avg) + '\nF: ' + str(f_measure_avg) + '\n')
