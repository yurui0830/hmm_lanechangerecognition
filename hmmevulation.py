from hmmtest import hmm_trian_test
from concatenate_features import concatenate_feature_label, feat_clip
import numpy as np
import matplotlib.pyplot as plt


def data_segmentation(classname, n_feature: int=6):
    # extract features
    (features, label, seq_range) = concatenate_feature_label(classname, n_feature=n_feature)
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


def cross_validation(x, y, cv: int=5, n_feature: int=6):
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
        accuracy[i], conf_temp = \
            hmm_trian_test(x_train, y_train, x_test, y_test, n_feature=n_feature, window_size=window_size)
        confusion = confusion + conf_temp
    return accuracy, confusion


accu = np.zeros([30])
prec = np.zeros([30])
recall = np.zeros([30])
for window_size in range(20, 50):
    x_r, y_r = data_segmentation('rightlc', n_feature=3)
    x_l, y_l = data_segmentation('leftlc', n_feature=3)
    x_lk, y_lk = data_segmentation('lk', n_feature=3)
    x = np.concatenate((x_r, x_l, x_lk))
    y = np.concatenate((y_r, y_l, y_lk))
    acc, conf = cross_validation(x, y, cv=10, n_feature=3)
    accu[window_size-20] = acc.mean()
    prec[window_size-20] = conf.diagonal().sum()/(conf.diagonal().sum()+conf[0:1, 2].sum())
    recall[window_size-20] = conf.diagonal().sum()/(conf.diagonal().sum()+conf[2, 0:1].sum())
print(accu)
print(prec)
print(recall)
plt.plot(np.arange(1, 4, 0.1), prec, 'b', label='precision')
plt.plot(np.arange(1, 4, 0.1), recall, 'g', label='recall')
plt.ylim((0.8, 1))
plt.xlabel('Seconds after the Behavior Starts (second)')
plt.ylabel('Recognition Performance')
plt.grid(True)
plt.show()
