import numpy as np
from emissionprob import b_emission
from hmmlearn.hmm import GaussianHMM
from sklearn.externals import joblib

class HiddenMarkovModel(object):

    """
    A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition, Lawrence R. Rabiner

    :param
        n_state(int): how many states in this model, default = 3
        A(array: n_state * n_state): transition probability
        startprob(array: n_state,): pi
        mu(array: n_state * n_feature): average of Gaussian distribution (usage: mu[i,:])
        sigma(array: n_state * n_feature * n_feature): covariances of Gaussian distribution (usage: sigma[i,:,:])

    * * * - - - * * * - - - * * *

    forward_backward: find the probability of the observation is from this model
    :param
        obs(array: n_seq * n_feature): observation sequence (usage: obs[t,:])
        n_seq(int): time steps inside one sequence
    :return
        alpha(array: n_state * n_seq): forward variable
        beta(array: n_state * n_seq): backward variable
        prob(float): evaluated probability

    * * * - - - * * * - - - * * *

    predict: find the best matched states
    :return
        prob_: evaluated probability
        sequence(array: n_seq,): matched states

    * * * - - - * * * - - - * * *

    Baum_Welch:
    :param
        iter(int): iteration times
        convergence(float): convergence criterion

    """

    def __init__(self, n_feature: int, n_state: int=3):
        self.n_state = n_state
        self.A = np.empty((n_state, n_state))
        self.startprob = np.empty((n_state,))
        self.mu = np.empty((n_state, n_feature))
        self.sigma = np.empty((n_state, n_feature, n_feature))
        self.B = None

# forward algorithm, P262; backward algorithm, P263
    def forward_backward(self, obs):
        n_seq = np.size(obs, 0)
        # calculate emission probability
        # B(array: n_state * n_seq): emission probability(usage: B[i, t])
        self.B = b_emission(self.n_state, n_seq, obs, self.mu, self.sigma)
        # forward algorithm
        # Equation 19
        alpha = np.zeros((self.n_state, n_seq))
        alpha[:, 0] = self.startprob * self.B[:, 0]
        # Equation 20
        for t in range(1, n_seq):
            for n in range(self.n_state):
                alpha[n, t] = (np.dot(alpha[:, t-1], self.A[:, n])) * self.B[n, t]
        # Equation 21
        prob = np.sum(alpha[:, -1])
        # backward algorithm
        # Equation 24
        beta = np.zeros((self.n_state, n_seq))
        beta[:-1] = 1
        # Equation 25
        for t in range(0, n_seq-1):
            for n in range(self.n_state):
                beta[n, t] = np.sum(self.A[n, :] * self.B[:, t+1] * beta[:, t+1])
        return alpha, beta, prob

# Viterbi Algorithm, P264
# find a best matched sequence (emission probability B contains the observations)
    def predict(self, obs):
        n_seq = np.size(obs, 0)
        # calculate emission probability
        self.B = b_emission(self.n_state, n_seq, obs, self.mu, self.sigma)
        # calculate delta and psi
        # Equation 32
        delta = np.zeros((self.n_state, n_seq))
        psi = np.zeros((self.n_state, n_seq))
        delta[:,0] = np.dot(self.startprob, self.B[:,0])
        psi[:,0] = 0
        # Equation 33
        for t in range(1, n_seq):
            for n in range(self.n_state):
                delta[n,t] = np.max(delta[:,t-1] * self.A[:,n] * self.B[n,t])
                psi[n,t] = np.argmax(delta[:,t-1] * self.A[:,n])
        # find the best matched sequence and probability
        # Equation 34
        prob_ = np.max(delta[:,-1])
        sequence = np.zeros((n_seq,), dtype=int)
        sequence[-1] = np.argmax(delta[:,-1])
        # Equation 35
        for t in range(n_seq-2, -1, -1):
            sequence[t] = psi[sequence[t+1],t+1]
        return sequence

# Baum-Welch Algorithm, P264-265
    def baumwelch(self, obs, n_seq: int, iter: int=1000, convergence=0.01):
        """
            xi(3darray: n_seq-1 * n_states * n_states): probability of being in state Si at time t, and state Sj at time t+1
            gamma(array: n_state * n_seq): probability of being in state Si at time t
            sum_gamma(array: n_state,): expected number of transitions from Si
            sum_xi(array: n_state * n_state): expected number of transitions from Si to Sj

        """
        # calculate emission probability
        self.B = b_emission(self.n_state, n_seq, obs, self.mu, self.sigma)
        # call alpha, beta, Prob
        (alpha, beta, prob) = self.forward_backward(obs, n_seq)
        for iteration in range(iter):
            # calculate gamma, xi
            xi = np.zeros((n_seq-1, self.n_state, self.n_state))
            gamma = np.zeros((self.n_state, n_seq))
            sum_gamma = np.zeros(self.n_state)
            sum_xi = np.zeros((self.n_state, self.n_state))
            # Equation 37
            for t in range(n_seq-1):
                for i in range(self.n_state):
                    for j in range(self.n_state):
                        xi[t,i,j] = alpha[i,t]*self.A[i,j]*self.B[j,t+1]*beta[j,t+1]
                xi[t,:,:] = xi[t,:,:] / xi[t,:,:].sum()
            # Equation 38, the last column of gamma (at the last time step, the probability of being state Si)
            for i in range(self.n_state):
                for t in range(n_seq-1):
                    gamma[i,t] = np.sum(xi[t,i,:])
                gamma[i,-1] = np.sum(xi[-1,:,i])
            # update A, B (mu, sigma), startprob
            # Equation 40(a): update startprob
            self.startprob = gamma[:, 0]
            # Equation 40(b): update A
            for i in range(self.n_state):
                for j in range(self.n_state):
                    self.A[i,j] = np.sum(xi[:,i,j]) / np.sum(gamma[i,0:-2])
                # normalize A
                self.A[i,:] = self.A[i,:] / self.A[i,:].sum()
                # Tutorial EM: Equation 10: update mu and sigma
                temp = 0
                for t in range(n_seq):
                    temp = temp + gamma[i,t]*obs[j,:]
                self.mu[i,:] = temp / np.sum(gamma[i,:])
                temp = 0
                for t in range(n_seq):
                    temp = temp + gamma[i,t] * np.outer(obs[t,:]-self.mu[i,:], obs[t,:]-self.mu[i,:])
                self.sigma[i,:,:] = temp / np.sum(gamma[i,:])
            # convert all NUM to zeros
            self.A = np.nan_to_num(self.A)
            self.startprob = np.nan_to_num(self.startprob)
            self.mu = np.nan_to_num(self.mu)
            self.sigma = np.nan_to_num(self.sigma)
            # calculate new probability
            (alpha_new, beta_new, prob_new) = self.forward_backward(obs, n_seq)
            # converge condition
            if abs(prob_new - prob) <= convergence:
                break
            # update hyper-parameters
            elif prob_new < prob:
                continue
            else:
                prob, alpha, beta = prob_new, alpha_new, beta_new
        return self

# supervised learning
    def supervised_learn(self, features, label, seq_range):
        """
        :param
            features(array: tot_seq * n_feature): inputs(labelled training set)
            length(array: n_sequence,): each element = time steps inside each sequence (also called n_seq)
            label(array: tot_seq,): label by state (0,1,2,...)
        """
        # find how many states in the labelled data, how many features in the data,
        # tot_seq(int): how many time steps in the whole input features
        self.n_state = int(np.ptp(label)+1)
        tot_seq = np.size(features, 0)
        # initialize some temporary variables
        count_start = np.zeros((self.n_state, ))
        # check the label at the first time step of each sequence
        # count_start: how many times the first time step of the sequence is labelled by one certain state
        for seq in range(np.size(seq_range, 0)-1):
            i = label[seq_range[seq]]
            count_start[i] = count_start[i]+1
        # calculate startprob
        for i in range(self.n_state):
            self.startprob[i] = count_start[i]/count_start.sum()
        # initialize some temporary variables
        for i in range(self.n_state):
            globals()['s'+str(i)] = []
        xi = np.zeros((self.n_state, self.n_state), dtype=int)
        # xi: count the numbers of transitions from Si to Sj
        # globals()['s'+str(i)] (array: unknown * n_feature): store the observations that labelled by one certain state
        for seq in range(tot_seq):
            i = label[seq]
            globals()['s'+str(i)].append(features[seq])
            if seq < tot_seq-1 and seq+1 not in seq_range:
                j = label[seq + 1]
                xi[i, j] = xi[i, j] + 1
        # calculate A (transition probability)
        for i in range(self.n_state):
            for j in range(self.n_state):
                self.A[i, j] = xi[i, j]/xi[i].sum()
        # calculate mu, sigma
        for i in range(self.n_state):
            globals()['s'+str(i)] = np.array(globals()['s'+str(i)], dtype=float)
            self.mu[i] = np.mean(globals()['s'+str(i)], axis=0)
            self.sigma[i] = np.cov(globals()['s'+str(i)], rowvar=0)
        # use mu and sigma to replace B temporarily
        self.B = None
        return self

# score: give the log probability
    def score(self, obs):
        n_seq = np.size(obs, 0)
        # calculate emission probability
        # B(array: n_state * n_seq): emission probability(usage: B[i, t])
        self.B = b_emission(self.n_state, n_seq, obs, self.mu, self.sigma)
        # forward algorithm
        # Equation 19
        alpha = np.zeros((self.n_state, n_seq))
        alpha[:, 0] = self.startprob * self.B[:, 0]
        # Equation 20
        for t in range(1, n_seq):
            for n in range(self.n_state):
                alpha[n, t] = (np.dot(alpha[:, t-1], self.A[:, n])) * self.B[n, t]
        # Equation 21
        prob = np.sum(alpha[:, -1])
        if prob <= 0:
            prob = 1e-10
        score = np.log(prob)
        return score


def hmm_train(features, label, seq_range):

    """
    function hmmtrain(features, label, seq_range)

    :param
        features, label, seq_range: return values from readcsvdata.py/concatenate_features.py
    :return:
        new_model(GaussianHMM class): well-trained model

    """

# find total number of states
    n_state = int(np.ptp(label)+1)

# supervised learning
# model: class HiddenMarkovModel, calculate hyper parameters from labels
    model = HiddenMarkovModel(n_feature=np.size(features, 1), n_state=n_state)
    model.supervised_learn(features, label, seq_range)

# Gaussian HMM (unsupervised)
# new model: class GaussianHMM (from hmmlearn), feed the new model by calculated parameters
    new_model = GaussianHMM(n_components=np.max(label)+1, covariance_type='full')
    new_model.transmat_ = model.A
    new_model.startprob_ = model.startprob
    new_model.means_ = model.mu
    new_model.covars_ = model.sigma
    covars = np.zeros([model.sigma.shape[0], model.sigma.shape[1]])
    # when the covariance_type = 'diag'
    #for i in range(model.sigma.shape[0]):
    #    covars[i] = np.diag(model.sigma[i])
    #new_model.covars_ = covars

# return value
    return new_model
