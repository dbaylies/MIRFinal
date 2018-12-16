import numpy as np
import matplotlib.pyplot as pyp
import numpy.linalg as linalg

def detect_segmentation(chromagram, fs_chromagram):

    (num_notes,num_samps) = chromagram.shape

    # The default parameters here follow the suggestions of Serra et al.
    tau = 1
    look_back = 2 # seconds
    w = round(fs_chromagram * look_back)
    m = round(w/tau + 1)
    N = num_samps-w  # Length of new chroma matrix

    x_delay = np.zeros((num_notes*m, N))

    for embed in np.arange(m):
        x_delay[num_notes*embed:num_notes*(embed+1), :] = \
            chromagram[:, (w - tau * embed):(w - tau * embed + N)]

    # pyp.imshow(x_delay,aspect='auto',origin='lower')
    # pyp.show()

    ### Calculate recurrence plot ###

    # Create cosine similarity matrix for k-NN step

    # cos_sim = np.zeros((num_samps, num_samps))

    dot_prod = np.matmul(np.transpose(x_delay), x_delay)

    mags = linalg.norm(x_delay, axis=0)

    mags = np.matrix(mags)

    mag_matrix = np.matmul(np.transpose(mags), mags)

    cos_sim = np.divide(dot_prod, mag_matrix)

    return mag_matrix
