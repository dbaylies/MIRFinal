import numpy as np
import matplotlib.pyplot as pyp
import Chroma_Detection
import Segmentation_Detection
import os

filepaths = os.listdir('../Audio/01_-_Please_Please_Me/')

# filepath = '../Audio/01_-_Please_Please_Me/01_-_I_Saw_Her_Standing_There.wav'
# filepath = '../Audio/01_-_Please_Please_Me/02_-_Misery.wav'
# filepath = '../Audio/T08-violin.wav'

filepath = '../Audio/01_-_Please_Please_Me/' + filepaths[13]

chromagram, fs_chromagram = Chroma_Detection.get_chromagram(filepath)

recurrence_plot = Segmentation_Detection.detect_segmentation(chromagram,fs_chromagram)

pyp.imshow(recurrence_plot,aspect='auto',origin='lower')
pyp.show()