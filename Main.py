import numpy as np
import matplotlib.pyplot as pyp
import get_chromagram
import recurrence_matrix
import gaussian_matrix
import novelty
import timelag_matrix

import os

root = '../Audio/'
album_num = 2
song_num = 2

albums = os.listdir(root)

songs = os.listdir(root + albums[album_num])

filepath = root + albums[album_num] + '/' + songs[song_num]

chromagram, fs_chromagram = get_chromagram.get_chromagram(filepath)

N, recurrence = recurrence_matrix.recurrence_matrix(chromagram,fs_chromagram)

L = timelag_matrix.timelag_matrix(N, recurrence)

P = gaussian_matrix.gaussian_matrix(fs_chromagram, L)

c = novelty.novelty(P)

pyp.plot(c)
pyp.show()