import numpy as np
import matplotlib.pyplot as pyp


def timelag_matrix(N, recurrence):

    # Create circular time-lag matrix
    L = np.zeros((N, N))

    for i in np.arange(N):
        for j in np.arange(N):
            k = np.mod(i + j - 2, N)
            L[i, j] = recurrence[k, j]

    pyp.figure()
    pyp.imshow(L)
    pyp.title('Time-Lag Matrix')

    return L
