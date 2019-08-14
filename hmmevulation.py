from hmmtest import hmm_trian_test
from concatenate_features import concatenate_feature_label, feat_clip
import numpy as np


def data_segmentation(classname):
    # extract features
    (features, label, seq_range) = concatenate_feature_label(classname, n_feature=6)
    # the labels of each sample become a new feature
    features = np.hstack((features, np.reshape(label, (-1, 1))))
    # lane keeping set
    if classname == 'lk':
        # how many sequences inside this data set
        n_sequence = seq_range.shape[0] - 1
        features_new = np.zeros((20 * n_sequence, features.shape[1]))
        for i in range(n_sequence):
            features_new[20*i:20*(i+1)] = features[seq_range[i]:seq_range[i]+20]
        # initialize variables
        x = np.zeros((n_sequence, 7 * 20))
        # generate labels (2 for lane keeping)
        y = np.full((n_sequence,), 2, dtype=int)
        for i in range(n_sequence):
            x[i] = np.reshape(features_new[20 * i:20 * (i + 1)], (-1))
        return x, y
    # left/right lane change set
    # how many sequences inside this data set
    n_sequence = seq_range.shape[0] - 1
    # initialize variables
    start_point = np.zeros((n_sequence,), dtype=int)
    features_new = np.zeros((20 * n_sequence, features.shape[1]))
    # find start point, extract part of the features and labels, update new seq_range
    for i in range(n_sequence):
        start = seq_range[i]
        stop = seq_range[i + 1]
        # find the start point by labels
        start_point[i] = np.argwhere(label[start:stop] == 1)[0] + seq_range[i]
        # extract features and labels (1 sec before and 1 sec after the maneuver begins)
        features_new[20*i:20*(i+1)] = features[start_point[i]-10:start_point[i]+10]
    # initialize variables
    x = np.zeros((n_sequence, 7*20))
    # generate labels (0 for rightlc, 1 for leftlc)
    if classname == 'leftlc':
        y = np.ones((n_sequence,), dtype=int)
    elif classname == 'rightlc':
        y = np.zeros((n_sequence,), dtype=int)
    for i in range(n_sequence):
        x[i] = np.reshape(features_new[20*i:20*(i+1)], (-1))
    # return value
    return x, y


def cross_validation(x, y, cv: int=5):
    # n_sampleL how many samples in the dataset; n_sample_fold: how many samples in each fold
    n_sample = x.shape[0]
    n_sample_fold = n_sample//cv
    # sample_list: use a number to label each sample
    sample_list = np.arange(0, n_sample)
    fold_old = np.arange(0, n_sample)
    # acc
    accuracy = np.zeros([cv, ])
    confusion = np.zeros([3, 3])
    for i in range(cv):
        # fold_old: samples which have not been selected for the test set
        # fold: samples in testing set during this fold
        # fold_train: samples in training set during this fold
        if i < cv-1:
            fold = np.random.choice(fold_old, n_sample_fold, replace=False)
        else:
            fold = fold_old
        fold_train = np.setdiff1d(sample_list, fold)
        # extract training and testing set
        x_test = x[fold]
        y_test = y[fold]
        x_train = x[fold_train]
        y_train = y[fold_train]
        # update fold_old, remove the samples used during this fold
        fold_old = np.setdiff1d(fold_old, fold)
        accuracy[i], conf_temp = hmm_trian_test(x_train, y_train, x_test, y_test)
        confusion = confusion + conf_temp
    return accuracy, confusion


x_r, y_r = data_segmentation('rightlc')
x_l, y_l = data_segmentation('leftlc')
x_lk, y_lk = data_segmentation('lk')
x = np.concatenate((x_r, x_l, x_lk))
y = np.concatenate((y_r, y_l, y_lk))
print(cross_validation(x, y, cv=10))
