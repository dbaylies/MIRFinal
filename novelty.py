import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as pyp


def novelty(P):

    # Get novelty curve
    c = linalg.norm(np.diff(P), axis=1)

    # normalize
    c = (c - np.min(c))
    c = c/np.max(c)

    pyp.imshow(np.diff(P))
    pyp.show()

    return c