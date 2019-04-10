import numpy as np
import scipy.stats

def b_emission(n_state, n_seq, obs, mu, sigma):

    """
    Tutorial EM:
    A Gentle Tutorial of the EM Algorithm and its Application to Parameter Estimation for Gaussian Mixture and
Hidden Markov Models, Jeff A. Bilmes

    :param
        n_state(int): number of states in one model
        n_seq(int): number of time steps in one sequence
        obs(array: n_seq * n_feature): observations(features)
        mu(array: n_state * n_feature): average
        sigma(3darray: n_state * n_feature * n_feature): covariance
    :return:
        b_prob(array: n_state * n_seq): emission probability: probability of state is Si with observation Ot

    """

# initialize emission probability B
    b_prob = np.zeros((n_state, n_seq))
    for state in range(n_state):
        # Gaussian Distribution model
        gaussian = scipy.stats.multivariate_normal(mean=mu[state,:], cov=sigma[state,:,:], allow_singular='True')
        for seq in range(n_seq):
            # obs_step(array: n_feature,): features in one single time step
            obs_step = obs[seq,:]
            # emission probability, Tutorial EM P8
            b_prob[state,seq] = gaussian.pdf(obs_step)

# normalize the emission probability to (0,1) for each sequence
    # when this model contains more than one state, the probability has to be normalized
    if n_state > 1:
        for seq in range(n_seq):
            b_prob[:,seq] = b_prob[:,seq] / b_prob[:,seq].sum()
    # when this model only contains one state, the probability will not be normalized
    # the probability which is larger than one will be set to one
    elif n_state == 1:
        for seq in range(n_seq):
            if b_prob[0, seq] > 1:
                b_prob[0, seq] = 1

# return value
    return b_prob
