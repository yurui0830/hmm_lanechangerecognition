import numpy as np
import pickle
from hmmlearn.hmm import GaussianHMM
from model import HiddenMarkovModel
from readcsvdata import generate_feature_label
from sklearn.externals import joblib


def hmm_train(features, label, seq_range):

    """
    function hmmtrain(features, label, seq_range)

    :param
        features, label, seq_range: return values from readcsvdata.py/concatenate_features.py
    :return:
        new_model(GaussianHMM class): well-trained model

    """

# find total number of states
    n_state = np.ptp(label)+1

# supervised learning
# model: class HiddenMarkovModel, calculate hyper parameters from labels
    model = HiddenMarkovModel(n_feature=np.size(features,1), n_state=n_state)
    model.supervised_learn(features, label, seq_range)

# Gaussian HMM (unsupervised)
# new model: class GaussianHMM (from hmmlearn), feed the new model by calculated parameters
    new_model = GaussianHMM(n_components=np.max(label)+1, covariance_type='full')
    new_model.transmat_ = model.A
    new_model.startprob_ = model.startprob
    new_model.means_ = model.mu
    new_model.covars_ = model.sigma

# return value
    return new_model
