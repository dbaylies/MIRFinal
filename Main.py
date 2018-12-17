import numpy as np
import matplotlib.pyplot as pyp
import Chroma_Detection
import Segmentation_Detection
import os

root = '../Audio/'
album_num = 2
song_num = 2

albums = os.listdir(root)

songs = os.listdir(root + albums[album_num])

filepath = root + albums[album_num] + '/' + songs[song_num]

chromagram, fs_chromagram = Chroma_Detection.get_chromagram(filepath)

recurrence_plot = Segmentation_Detection.detect_segmentation(chromagram,fs_chromagram)

pyp.imshow(recurrence_plot)
pyp.title('Recurrence Plot for ' + songs[song_num])
pyp.show()