import numpy as np
from readcsvdata import generate_feature_label
from extractfeat import feat_extract
from hmmtrain import hmm_train


# extract training set from 101 data set
(features_r, label_r, seq_range_r) = generate_feature_label('rightlc_101', n_feature=3)
(features_l, label_l, seq_range_l) = generate_feature_label('leftlc_101', n_feature=3)
(features_lk, label_lk, seq_range_lk) = generate_feature_label('lk_101', n_feature=3)

# extract features (training the model using extracted features)
(new_features_r, new_label_r, new_seq_range_r) = feat_extract(features_r, label_r, seq_range_r)
(new_features_l, new_label_l, new_seq_range_l) = feat_extract(features_l, label_l, seq_range_l)


# train the model by 101 data set
model_right = hmm_train(new_features_r, new_label_r, new_seq_range_r)
model_left = hmm_train(new_features_l, new_label_l, new_seq_range_l)
model_lk = hmm_train(features_lk, label_lk, seq_range_lk)

# test the model by new data set
(features, label, seq_range) = generate_feature_label('leftlc_i80', n_feature=3)
# find start point for each lane change behavior in the testing set
start_point = np.zeros((np.size(seq_range, 0) - 1,), dtype=int)
# initialize start_point(array: n_sequence,): store index of each start point

# right and left class

for i in range(np.size(seq_range, 0) - 1):
    start = seq_range[i]
    stop = seq_range[i + 1]
    # find the start point by labels
    start_point[i] = np.argwhere(label[start:stop] == 1)[0] + seq_range[i]
"""
# lk class
for i in range(np.size(seq_range, 0) - 1):
    start_point[i] = seq_range[i]+30
"""

# result_prob(array: number of sequences * n_state): store probabilities from each model
# result(array: number of sequences,): store the final result (state)
result_prob = np.zeros((np.size(seq_range, 0)-1, 3))
result = np.zeros((np.size(seq_range, 0)-1,))
for i in range(np.size(seq_range, 0)-1):
    """
    # whole process
    prob_r = model_right.score(features[seq_range[i]:seq_range[i+1]])
    prob_l = model_left.score(features[seq_range[i]:seq_range[i+1]])
    prob_lk = model_lk.score(features[seq_range[i]:seq_range[i+1]])
    """
    # 2s before + 1s after
    if start_point[i] - 20 > seq_range[i]:
        start = start_point[i] - 20
    else:
        start = seq_range[i]
    stop = start_point[i] + 10
    prob_r = model_right.score(features[start:stop])
    prob_l = model_left.score(features[start:stop])
    prob_lk = model_lk.score(features[start:stop])
    """
    # 2.5s before + o.5s after
    if start_point[i] - 25 > seq_range[i]:
        start = start_point[i] - 25
    else:
        start = seq_range[i]
    stop = start_point[i] + 5
    prob_r = model_right.score(features[start:stop])
    prob_l = model_left.score(features[start:stop])
    prob_lk = model_lk.score(features[start:stop])
    """
    # store and compare probabilities
    result_prob[i] = [prob_r, prob_l, prob_lk]
    result[i] = np.argmax(result_prob[i])

print(np.count_nonzero(result == 1)/(np.size(seq_range, 0)-1))
