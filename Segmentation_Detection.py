import numpy as np
import matplotlib.pyplot as pyp
import numpy.linalg as linalg
import scipy.signal as signal

def detect_segmentation(chromagram, fs_chromagram):

    # Parameters
    kappa = 0.1
    # Gaussian kernel lengths
    s_l_sec = 0.3 # seconds
    s_t_sec = 15 # Seconds
    variance = 0.16

    (num_notes, num_samps) = chromagram.shape

    # The default parameters here follow the suggestions of Serra et al.
    tau = 1
    look_back = 2  # seconds
    w = round(fs_chromagram * look_back)
    m = round(w/tau + 1)
    N = num_samps-w  # Length of new chroma matrix

    x_delay = np.zeros((num_notes*m, N))

    for embed in np.arange(m):
        x_delay[num_notes*embed:num_notes*(embed+1), :] = \
            chromagram[:, (w - tau * embed):(w - tau * embed + N)]

    # pyp.imshow(x_delay,aspect='auto',origin='lower')
    # pyp.show()

    #################################
    ### Calculate recurrence plot ###
    #################################

    # Create cosine similarity matrix for k-NN step
    # TODO: implement Serra et al.'s method instead (norm of difference between vectors)
    # and compare with cosine similarity

    dot_prod = np.matmul(np.transpose(x_delay), x_delay)

    mags = linalg.norm(x_delay, axis=0)

    mags = np.matrix(mags)

    mag_matrix = np.matmul(np.transpose(mags), mags)

    cos_sim = np.divide(dot_prod, mag_matrix, out=np.zeros_like(mag_matrix), where=mag_matrix != 0)

    # find nearest neighbors
    recurrence = np.zeros((N, N))

    K = round(kappa * N)

    neighbors = np.zeros((K, N))

    # Matrix with indices of K nearest neighbors for each sample
    for i in np.arange(N):
        neighbors[:, i] = np.argsort(abs(cos_sim[:, i]), axis=0)[-K:].flatten()

    # for i in np.arange(N):
    #     for j in np.arange(N):
    #         if j in neighbors[:, i] and i in neighbors[:, j]:
    #             recurrence[i, j] = 1

    #################################
    ###  Get structural features  ###
    #################################

    # Create circular time-lag matrix

    L = np.zeros((N,N))

    # for i in np.arange(N):
    #     for j in np.arange(N):
    #         k = np.mod(i + j - 2, N)
    #         L[i,j] = recurrence[k,j]

    # Create bi-variate rectangular Gaussian kernel

    s_l = round(s_l_sec * fs_chromagram)
    s_t = round(s_t_sec * fs_chromagram)

    std_dev = np.sqrt(variance)

    g_l = np.matrix(signal.gaussian(s_l,std_dev))
    g_t = np.matrix(signal.gaussian(s_t,std_dev))

    G = np.matmul(np.transpose(g_l),g_t)

    return G
