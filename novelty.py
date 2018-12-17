import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as pyp
import scipy.signal as signal


def novelty(P):

    # Get novelty curve
    c = linalg.norm(np.diff(P), axis=0)

    # normalize
    c = (c - np.amin(c))
    c = c/np.amax(c)

    # c = signal.medfilt(c,33)

    pyp.figure()
    pyp.plot(c)
    pyp.title('novelty curve')

    return c
