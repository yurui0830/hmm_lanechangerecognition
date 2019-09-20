import numpy as np
from hmmevulation import data_segmentation
from hmmtest import hmm_trian_test
import matplotlib.pyplot as plt


"""
generalization test

trained on one 

x_r, y_r = data_segmentation('rightlc', n_feature=3, dataset='101', window_size=20)
x_l, y_l = data_segmentation('leftlc', n_feature=3, dataset='101', window_size=20)
x_lk, y_lk = data_segmentation('lk', n_feature=3, dataset='101', window_size=20)
train_x = np.concatenate((x_r, x_l, x_lk))
train_y = np.concatenate((y_r, y_l, y_lk))

x_r_test, y_r_test = data_segmentation('rightlc', n_feature=3, dataset='i80', window_size=20)
x_l_test, y_l_test = data_segmentation('leftlc', n_feature=3, dataset='i80', window_size=20)
x_lk_test, y_lk_test = data_segmentation('lk', n_feature=3, dataset='i80', window_size=20)
test_x = np.concatenate((x_r_test, x_l_test, x_lk_test))
test_y = np.concatenate((y_r_test, y_l_test, y_lk_test))

acc, confusion = hmm_trian_test(train_x, train_y, test_x, test_y, n_feature=3, window_size=20)

print(acc, confusion)

"""
conf = np.zeros([3, 3])
conf[0] = [65, 0, 10]
conf[1] = [0, 67, 12]
conf[2] = [3, 1, 46]
acc = conf.diagonal().sum()/conf.sum()
prec = conf.diagonal().sum()/(conf.diagonal().sum()+conf[0:1, 2].sum())
recall = conf.diagonal().sum()/(conf.diagonal().sum()+conf[2, 0:1].sum())
print(acc, prec, recall)
