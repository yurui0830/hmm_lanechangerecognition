from hmmtest import hmm_trian_test
from concatenate_features import concatenate_feature_label, feature_label, feat_clip
import numpy as np
import matplotlib.pyplot as plt


def data_segmentation(classname, n_feature: int=6, dataset='whole', window_size: int=20):
    """
    usage: x, y = data_segmentation('rightlc', n_feature=3)

    :param
        classname(string: 'rightlc', 'leftlc' or 'lk'):
        n_feature(int: 3 or 6):
        dataset(string: 'whole', '101' or 'i80'):
        window_size
    :return:
        x(array: n_sequence*?): sequential data(all features + states)
        y(array: n_sequence,): class label
    """
    # extract features
    if dataset == 'whole':
        (features, label, seq_range) = concatenate_feature_label(classname, n_feature=n_feature)
    else:
        (features, label, seq_range) = feature_label(classname, dataset,  n_feature=n_feature)
    # the labels of each sample become a new feature
    features = np.hstack((features, np.reshape(label, (-1, 1))))
    # lane keeping set
    if classname == 'lk':
        # how many sequences inside this data set
        n_sequence = seq_range.shape[0] - 1
        features_new = np.zeros((window_size * n_sequence, features.shape[1]))
        for i in range(n_sequence):
            features_new[window_size*i:window_size*(i+1)] = features[seq_range[i]:seq_range[i]+window_size]
        # initialize variables
        x = np.zeros((n_sequence, (n_feature+1) * window_size))
        # generate labels (2 for lane keeping)
        y = np.full((n_sequence,), 2, dtype=int)
        for i in range(n_sequence):
            x[i] = np.reshape(features_new[window_size * i:window_size * (i + 1)], (-1))
        return x, y
    # left/right lane change set
    # how many sequences inside this data set
    n_sequence = seq_range.shape[0] - 1
    # initialize variables
    start_point = np.zeros((n_sequence,), dtype=int)
    features_new = np.zeros((window_size * n_sequence, features.shape[1]))
    # find start point, extract part of the features and labels, update new seq_range
    for i in range(n_sequence):
        start = seq_range[i]
        stop = seq_range[i + 1]
        # find the start point by labels
        start_point[i] = np.argwhere(label[start:stop] == 1)[0] + seq_range[i]
        # extract features and labels (1 sec before and 1 sec after the maneuver begins)
        features_new[window_size*i:window_size*(i+1)] = features[start_point[i]-10:start_point[i]+window_size-10]
    # initialize variables
    x = np.zeros((n_sequence, (n_feature+1)*window_size))
    # generate labels (0 for rightlc, 1 for leftlc)
    if classname == 'leftlc':
        y = np.ones((n_sequence,), dtype=int)
    elif classname == 'rightlc':
        y = np.zeros((n_sequence,), dtype=int)
    for i in range(n_sequence):
        x[i] = np.reshape(features_new[window_size*i:window_size*(i+1)], (-1))
    # return value
    return x, y


def single_test(x, y, n_feature: int=6):
    accuracy = np.zeros((20,))
    for i in range(1, 20):
        # extract training and testing set
        x_test = x[0]
        y_test = y[0]
        x_train = x[1:]
        y_train = y[1:]
        accuracy[i], _ = hmm_trian_test(x_train, y_train, x_test[0:4*i], y_test, n_feature=n_feature)
    return accuracy


x_r, y_r = data_segmentation('rightlc', n_feature=3)
x_l, y_l = data_segmentation('leftlc', n_feature=3)
x_lk, y_lk = data_segmentation('lk', n_feature=3)
x = np.concatenate((x_r, x_l, x_lk))
y = np.concatenate((y_r, y_l, y_lk))
acc = single_test(x, y, n_feature=3)
print('accuracy:', acc)
