import numpy as np
from readcsvdata import generate_feature_label
from concatenate_features import concatenate_feature_label
from model import HiddenMarkovModel
from hmmtrain import hmm_train
from extractfeat import feat_clip

def hmm_test(testclass, training_set = 'part', slot = '1s', n_feature: int = 6):

    """
    funtion hmm_test(n_feature)

    :param
        testfile(str: 'rightlc', 'leftlc' or 'lk'): name of test class
        training_set(str: 'whole', or 'part'): train the model by the complete or part of the dataset, default: part
        slot(str: '1s' or '0.5s'): 1s or 0.5s, default: 1s. useless if training set is 'whole'
        n_feature(int: 2, 3 or 6): how many features are kept, default: 6

    :return
        precision(float): true positive rate
    """

# extract training set
    (features_r, label_r, seq_range_r) = concatenate_feature_label('rightlc', n_feature=n_feature)
    (features_l, label_l, seq_range_l) = concatenate_feature_label('leftlc', n_feature=n_feature)
    (features_lk, label_lk, seq_range_lk) = concatenate_feature_label('lk', n_feature=n_feature)

# extract testing set, cross validation
    # n_test(int): how many samples in one testing set
    # temporary variable: temporarily store all the samples of test class
    if testclass == 'rightlc':
        state = 0
        n_test = round(np.size(seq_range_r, 0) / 3)
        temp_fea = features_r
        temp_lab = label_r
        temp_sr = seq_range_r
        fea = 'features_r'
        lab = 'label_r'
        sr = 'seq_range_r'
    elif testclass == 'leftlc':
        state = 1
        n_test = round(np.size(seq_range_l, 0) / 3)
        temp_fea = features_l
        temp_lab = label_l
        temp_sr = seq_range_l
        fea = 'features_l'
        lab = 'label_l'
        sr = 'seq_range_l'
    elif testclass == 'lk':
        state = 2
        n_test = round(np.size(seq_range_lk, 0) / 3)
        temp_fea = features_lk
        temp_lab = label_lk
        temp_sr = seq_range_lk
        fea = 'features_lk'
        lab = 'label_lk'
        sr = 'seq_range_lk'
    # split test set, cross validation
    # tp: true positive rate of each round
    tpr = np.zeros((3,))
    # rd: round
    for rd in range(3):
        # generate testing set (features, label, seq_range)
        if rd == 0:
            start = n_test*rd
            stop = n_test*rd + n_test
            features = temp_fea[temp_sr[start]:temp_sr[stop]]
            label = temp_lab[temp_sr[start]:temp_sr[stop]]
            seq_range = temp_sr[start:stop] - temp_sr[start]
            globals()[sr] = np.delete(temp_sr, np.s_[start:stop]) - temp_sr[stop]
        elif rd == 1:
            start = n_test * rd
            stop = n_test * rd + n_test
            features = temp_fea[temp_sr[start]:temp_sr[stop]]
            label = temp_lab[temp_sr[start]:temp_sr[stop]]
            seq_range = temp_sr[start:stop] - temp_sr[start]
            globals()[sr] = np.delete(temp_sr, np.s_[start:stop])
            globals()[sr][stop:] = globals()[sr][stop:] - (temp_sr[stop]-temp_sr[start])
        elif rd == 2:
            start = n_test*rd
            stop = -1
            features = temp_fea[temp_sr[start]:temp_sr[stop]]
            label = temp_lab[temp_sr[start]:temp_sr[stop]]
            seq_range = temp_sr[start:stop] - temp_sr[start]
            globals()[sr] = np.delete(temp_sr, np.s_[start:stop])
        # features_r, label_r, seq_range_r: training set
        globals()[fea] = np.delete(temp_fea, np.s_[start:stop], 0)
        globals()[lab] = np.delete(temp_lab, np.s_[start:stop], 0)

# train three HMMs with labelled data
        if training_set == 'whole':
            slot == 'whole'
            model_right = hmm_train(features_r, label_r, seq_range_r)
            model_left = hmm_train(features_l, label_l, seq_range_l)
        elif training_set == 'part':
            # extract features (training the model using extracted features)
            (part_features_r, part_label_r, part_seq_range_r) = feat_clip(features_r, label_r, seq_range_r)
            (part_features_l, part_label_l, part_seq_range_l) = feat_clip(features_l, label_l, seq_range_l)
            # train the model by partial sequence
            model_right = hmm_train(part_features_r, part_label_r, part_seq_range_r)
            model_left = hmm_train(part_features_l, part_label_l, part_seq_range_l)
        model_lk = hmm_train(features_lk, label_lk, seq_range_lk)

# find start point for each lane change behavior in the testing set
        start_point = np.zeros((np.size(seq_range, 0) - 1,), dtype=int)
        if testclass != 'lk':
            # initialize start_point(array: n_sequence,): store index of each start point
            for i in range(np.size(seq_range, 0) - 1):
                start = seq_range[i]
                stop = seq_range[i + 1]
                # find the start point by labels
                start_point[i] = np.argwhere(label[start:stop] == 1)[0] + seq_range[i]
        else:
            for i in range(np.size(seq_range, 0) - 1):
                start_point[i] = seq_range[i]+30

# test: compare probabilities
        # result_prob(array: number of sequences * n_state): store probabilities from each model
        # result(array: number of sequences,): store the final result (state)
        result_prob = np.zeros((np.size(seq_range, 0)-1, 3))
        result = np.zeros((np.size(seq_range, 0)-1,))
        for i in range(np.size(seq_range, 0)-1):
            # whole process
            if slot == 'whole':
                prob_r = model_right.score(features[seq_range[i]:seq_range[i+1]])
                prob_l = model_left.score(features[seq_range[i]:seq_range[i+1]])
                prob_lk = model_lk.score(features[seq_range[i]:seq_range[i+1]])
            # 2s before + 1s after
            elif slot == '1s':
                if start_point[i] - 20 > seq_range[i]:
                    start = start_point[i] - 20
                else:
                    start = seq_range[i]
                stop = start_point[i] + 10
                prob_r = model_right.score(features[start:stop])
                prob_l = model_left.score(features[start:stop])
                prob_lk = model_lk.score(features[start:stop])
            # 2.5s before + o.5s after
            elif slot == '0.5s':
                if start_point[i] - 25 > seq_range[i]:
                    start = start_point[i] - 25
                else:
                    start = seq_range[i]
                stop = start_point[i] + 5
                prob_r = model_right.score(features[start:stop])
                prob_l = model_left.score(features[start:stop])
                prob_lk = model_lk.score(features[start:stop])
            # store and compare probabilities
            result_prob[i] = [prob_r, prob_l, prob_lk]
            result[i] = np.argmax(result_prob[i])

# print specific results
            #print(testclass, rd, '-fold:')
            #print(result_prob[i])
            #print(result[i])

# calculate true positive rate, cross validation result
        # tpr(array: rd,): true positive rate
        tpr[rd] = np.count_nonzero(result == state)/(np.size(seq_range, 0)-1)
    # calculate average of true positive rates
    precision = np.mean(tpr)

# return value
    return precision
