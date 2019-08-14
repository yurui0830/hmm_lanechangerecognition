import numpy as np
from concatenate_features import concatenate_feature_label
from extractfeat import feat_clip
from hmm_model import hmm_train
from sklearn.model_selection import train_test_split


def unfold_x(x, y, dataset='train'):
    if dataset == 'test':
        feature = np.zeros((5000, 6))
        for i in range(x.shape[0]):
            temp = np.reshape(x[i], (20, 7))
            feature[i * 20:(i + 1) * 20] = temp[:, 0:6]
        return feature

    r, l, lk = [0] * 3
    feature_r = np.zeros((5000, 6))
    feature_l = np.zeros((5000, 6))
    feature_lk = np.zeros((5000, 6))
    label_r = np.zeros((5000,))
    label_l = np.zeros((5000,))
    label_lk = np.zeros((5000,))
    for i in range(x.shape[0]):
        if y[i] == 0:
            temp = np.reshape(x[i], (20, 7))
            feature_r[r * 20:(r + 1) * 20] = temp[:, 0:6]
            label_r[r * 20:(r + 1) * 20] = temp[:, 6]
            r = r + 1
        elif y[i] == 1:
            temp = np.reshape(x[i], (20, 7))
            feature_l[20 * l:20 * (l + 1)] = temp[:, 0:6]
            label_l[l * 20:(l + 1) * 20] = temp[:, 6]
            l = l + 1
        elif y[i] == 2:
            temp = np.reshape(x[i], (20, 7))
            feature_lk[20 * lk: 20 * (lk + 1)] = temp[:, 0:6]
            label_lk[lk * 20:(lk + 1) * 20] = temp[:, 6]
            lk = lk + 1

    seq_range_r = np.arange(0, (r + 1) * 20, 20)
    seq_range_l = np.arange(0, (l + 1) * 20, 20)
    seq_range_lk = np.arange(0, (lk + 1) * 20, 20)
    feature_r = np.delete(feature_r, np.s_[r * 20:5000], axis=0)
    label_r = np.delete(label_r, np.s_[r * 20:5000])
    feature_l = np.delete(feature_l, np.s_[l * 20:5000], axis=0)
    label_l = np.delete(label_l, np.s_[l * 20:5000])
    feature_lk = np.delete(feature_lk, np.s_[lk * 20:5000], axis=0)
    label_lk = np.delete(label_lk, np.s_[lk * 20:5000])

    return feature_r, label_r, seq_range_r, feature_l, label_l, seq_range_l, feature_lk, label_lk, seq_range_lk

def hmm_trian_test(trian_x, train_y, test_x, test_y):

    """
    funtion hmm_test(n_feature)

    :param
        training_set(str: 'whole', or '1s'): train the model by the complete or part of the dataset, default: 1s
        n_feature(int: 2, 3 or 6): how many features are kept, default: 6

    :return
        precision(float): true positive rate
    """

    # prepare training set
    feat_r, label_r, s_range_r, feat_l, label_l, s_range_l, feat_lk, label_lk, s_range_lk = unfold_x(trian_x, train_y)
    # model training
    model_right = hmm_train(feat_r, label_r, s_range_r)
    model_left = hmm_train(feat_l, label_l, s_range_l)
    model_lk = hmm_train(feat_lk, label_lk, s_range_lk)

    # prepare testing set
    n_sequence = test_x.shape[0]
    test_feat = unfold_x(test_x, test_y, dataset='test')
    seq_range = np.arange(0, (n_sequence+1)*20, 20)
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
    for i in range(n_sequence):
        confusion[result[i], test_y[i]] = confusion[result[i], test_y[i]] + 1
    # acc(array: rd,): true positive rate
    acc = np.count_nonzero(result == test_y)/n_sequence

# return value
    return acc, confusion
