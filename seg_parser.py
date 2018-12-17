import numpy as np
import pandas as pd
import os

def seg_parse(album_num):

    root = '../SegData/'

    albums = os.listdir(root)
    files = os.listdir(root + albums[album_num])

    # Initialize list
    new_files = []

    # Initialize dictionary
    seg_times = {}

    for i in np.arange(len(files)):
        if files[i][-4:] == '.lab':
            seg_times[files[i]] = pd.read_csv(root + albums[album_num] + '/' + files[i], delimiter='\t').iloc[:, 0]
            new_files.append(files[i])
            i += 1

    return seg_times, new_files
