import numpy as np

def feat_extract(features, label, seq_range):

    """
    funtion feat_extract (features, label, seq_range)
    :param
        features(array: n_seq * n_feature): result from readcsvdata.py
        label(array: n_seq,): result from readcsvdata.py
        seq_range(array: n_sequence+1, ): result from readcsvdata.py
    :return:
        features_new(array: n_seq_new * n_feature): 20s before the start point and 30s after the start point
        label_new(array: n_seq_new,): 20s before the start point and 30s after the start point
        seq_range_new(array: n_sequence+1,): 20s before the start point and 30s after the start point
    """

    # how many sequences inside this data set
    n_sequence = np.size(seq_range, 0) - 1

# initialize variables
    start_point = np.zeros((n_sequence,), dtype=int)
    seq_range_new = np.zeros((n_sequence+1,), dtype=int)
    features_new = np.zeros((50 * n_sequence, np.size(features, 1)))
    label_new = np.zeros((50 * n_sequence,), dtype=int)

# find start point, extract part of the features and labels, update new seq_range
    for i in range(n_sequence):
        start = seq_range[i]
        stop = seq_range[i + 1]
        # find the start point by labels
        start_point[i] = np.argwhere(label[start:stop] == 1)[0] + seq_range[i]
        # extract features and labels
        if start_point[i]-20 > seq_range[i]:
            seq_range_new[i+1] = seq_range_new[i] + 50
            features_new[seq_range_new[i]:seq_range_new[i+1]] = features[start_point[i]-20:start_point[i]+30]
            label_new[seq_range_new[i]:seq_range_new[i+1]] = label[start_point[i]-20:start_point[i]+30]
        else:
            seq_range_new[i+1] = seq_range_new[i] + start_point[i]+30 - seq_range[i]
            features_new[seq_range_new[i]:seq_range_new[i+1]] = features[seq_range[i]:start_point[i]+30]
            label_new[seq_range_new[i]:seq_range_new[i+1]] = label[seq_range[i]:start_point[i]+30]
    # delete unused rows from features_new and label_new
    features_new = np.delete(features_new, np.s_[seq_range_new[n_sequence]:], 0)
    label_new = np.delete(label_new, np.s_[seq_range_new[n_sequence]:], 0)

# return value
    return features_new, label_new, seq_range_new
