import numpy as np
from readcsvdata import generate_feature_label
from concatenate_features import concatenate_feature_label
from model import HiddenMarkovModel
from hmmtrain import hmm_train
from extractfeat import feat_clip
import matplotlib.pyplot as plt

#(features_1, label_1, seq_range_1) = generate_feature_label('leftlc_101', n_feature=3)
#(features_2, label_2, seq_range_2) = generate_feature_label('leftlc_i80', n_feature=3)
(features, label, seq_range) = concatenate_feature_label('leftlc', n_feature=3, satruation=False)

plt.plot(features[seq_range[1]:seq_range[2],1])
plt.ylabel('heading angles')
plt.show()

#(new_features_1, new_label_1, new_seq_range_1) = feat_clip(features_1, label_1, seq_range_1)
#(new_features_2, new_label_2, new_seq_range_2) = feat_clip(features_2, label_2, seq_range_2)
#(new_features, new_label, new_seq_range) = feat_clip(features, label, seq_range)

#left = hmm_train(new_features, new_label, new_seq_range)
#left_1 = hmm_train(new_features_1, new_label_1, new_seq_range_1)
#left_2 = hmm_train(new_features_2, new_label_2, new_seq_range_2)

#print(left.means_)
#print(left_1.means_)
#print(left_2.means_)
