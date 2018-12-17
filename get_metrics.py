import numpy as np

def get_metrics(seg_times, files, onset_t):

    f_plus = 0
    f_minus = 0
    correct = 0

    seg_indices = np.zeros(len(seg_times))
    f_plus_flag = 0

    for i in np.arange(len(onset_t)):
        diffs = abs(seg_times - onset_t[i])
        for j in np.arange(len(diffs)):
            if diffs[j] < 7:
                correct += 1
                # Mark as found
                seg_indices[j] = 1
                f_plus_flag = 1

        if not f_plus_flag:
            f_plus += 1

        f_plus_flag = 0

    for i in np.arange(len(seg_indices)):
        if seg_indices[i] == 0:
            f_minus += 1

    # Ok if this is bad...
    precision = correct/(correct+f_plus)

    # Want minimal false negative! Don't want to miss an entire segment...
    recall = correct/(correct+f_minus)
    f_measure = (2*precision*recall)/(precision+recall)

    return precision, recall, f_measure