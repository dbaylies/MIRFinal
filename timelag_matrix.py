import numpy as np


def timelag_matrix(N, recurrence):

    # Create circular time-lag matrix
    L = np.zeros((N,N))

    for i in np.arange(N):
        for j in np.arange(N):
            k = np.mod(i + j - 2, N)
            L[i,j] = recurrence[k,j]

    return L