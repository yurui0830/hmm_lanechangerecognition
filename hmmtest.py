import numpy as np
from concatenate_features import concatenate_feature_label
from hmm_model import hmm_train
from sklearn.model_selection import train_test_split


def unfold_x(x, y, n_feature: int=6, dataset='train', window_size: int=20):
    if dataset == 'test':
        feature = np.zeros((20000, n_feature))
        if x.ndim == 1:
            temp = np.reshape(x, (-1, n_feature + 1))
            return temp[:, 0:n_feature]
        for i in range(x.shape[0]):
            temp = np.reshape(x[i], (window_size, n_feature+1))
            feature[i * window_size:(i + 1) * window_size] = temp[:, 0:n_feature]
        return feature

    r, l, lk = [0] * 3
    feature_r = np.zeros((20000, n_feature))
    feature_l = np.zeros((20000, n_feature))
    feature_lk = np.zeros((20000, n_feature))
    label_r = np.zeros((20000,), dtype=int)
    label_l = np.zeros((20000,), dtype=int)
    label_lk = np.zeros((20000,), dtype=int)
    for i in range(x.shape[0]):
        if y[i] == 0:
            temp = np.reshape(x[i], (window_size, n_feature+1))
            feature_r[r * window_size:(r + 1) * window_size] = temp[:, 0:-1]
            label_r[r * window_size:(r + 1) * window_size] = temp[:, -1]
            r = r + 1
        elif y[i] == 1:
            temp = np.reshape(x[i], (window_size, n_feature+1))
            feature_l[window_size * l:window_size * (l + 1)] = temp[:, 0:-1]
            label_l[l * window_size:(l + 1) * window_size] = temp[:, -1]
            l = l + 1
        elif y[i] == 2:
            temp = np.reshape(x[i], (window_size, n_feature+1))
            feature_lk[window_size * lk: window_size * (lk + 1)] = temp[:, 0:-1]
            label_lk[lk * window_size:(lk + 1) * window_size] = temp[:, -1]
            lk = lk + 1

    seq_range_r = np.arange(0, (r + 1) * window_size, window_size)
    seq_range_l = np.arange(0, (l + 1) * window_size, window_size)
    seq_range_lk = np.arange(0, (lk + 1) * window_size, window_size)
    feature_r = np.delete(feature_r, np.s_[r * window_size:20000], axis=0)
    label_r = np.delete(label_r, np.s_[r * window_size:20000])
    feature_l = np.delete(feature_l, np.s_[l * window_size:20000], axis=0)
    label_l = np.delete(label_l, np.s_[l * window_size:20000])
    feature_lk = np.delete(feature_lk, np.s_[lk * window_size:20000], axis=0)
    label_lk = np.delete(label_lk, np.s_[lk * window_size:20000])

    return feature_r, label_r, seq_range_r, feature_l, label_l, seq_range_l, feature_lk, label_lk, seq_range_lk

def hmm_trian_test(trian_x, train_y, test_x, test_y, n_feature: int=6, window_size: int=20):

    """
    usage: acc, confusion = hmm_train_test(trainX, trainY, testX, testY, n_feature=3)

    :param

        n_feature(int: 3 or 6): how many features are kept, default: 6

    :return
        acc(float): arruracy
        confusion(array: n_class*n_class): confusion matrix
    """

    # prepare training set
    feat_r, label_r, s_range_r, feat_l, label_l, s_range_l, feat_lk, label_lk, s_range_lk = \
        unfold_x(trian_x, train_y, n_feature, window_size=window_size)
    # model training
    model_right = hmm_train(feat_r, label_r, s_range_r)
    model_left = hmm_train(feat_l, label_l, s_range_l)
    model_lk = hmm_train(feat_lk, label_lk, s_range_lk)

    # prepare testing set

    test_feat = unfold_x(test_x, test_y, n_feature, dataset='test', window_size=window_size)
    if test_x.ndim == 1:
        n_sequence = 1
        seq_range = np.array([0, test_feat.shape[0]])
    else:
        n_sequence = test_x.shape[0]
        seq_range = np.arange(0, (n_sequence + 1) * window_size, window_size)
    # model testing
    result_prob = np.zeros((n_sequence, 3))
    result = np.zeros((n_sequence,), dtype=int)
    for i in range(n_sequence):
        prob_r = model_right.score(test_feat[seq_range[i]:seq_range[i+1]])
        prob_l = model_left.score(test_feat[seq_range[i]:seq_range[i+1]])
        prob_lk = model_lk.score(test_feat[seq_range[i]:seq_range[i+1]])
        # store and compare probabilities
        result_prob[i] = [prob_r, prob_l, prob_lk]
        result[i] = np.argmax(result_prob[i])

    # confusion matrix: row: predictions; column: labels
    confusion = np.zeros([3, 3])
    if n_sequence > 1:
        for i in range(n_sequence):
            confusion[result[i], test_y[i]] = confusion[result[i], test_y[i]] + 1
    # acc(array: rd,): true positive rate
    acc = np.count_nonzero(result == test_y)/n_sequence

# return value
    return acc, confusion
