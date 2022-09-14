import numpy as np
#import copy
from scipy.stats import norm


def discount(t):
    return 0.5 / np.power(t, 0.6)

class MultinomialHMMonline:

    def __init__(self, num_states, num_obs, probs_z=None, iid=False):
        ' implementation of the online EM algorithm in Mongillo & Deneve (2008), "Online Learning with HMMs" '

        self.num_states, self.num_obs = num_states, num_obs

        self.iid=iid

        self.theta = np.random.rand(num_states, num_obs) # emission probabilities; denoted 'b' in Mongillo & Deneve
        self.theta /= self.theta.sum(1, keepdims=True)
        #self.theta = np.ones((num_states, num_obs)) / num_states

        prob_change = 1 - 2/(num_states+1) # 0.3
        self.phi = (1 - prob_change) * np.eye(num_states) + prob_change * (1 - np.eye(num_states)) / (num_states - 1) # initial transition matrix estimate; denoted 'a' in Mongillo & Deneve
        #self.phi = np.ones((num_states, num_states)) / num_states
        if iid: self.phi = np.repeat(self.phi.sum(0, keepdims=True), num_states, 0)

        if probs_z is not None:
            self.probs_z = probs_z
        else:
            self.probs_z = np.ones(num_states) / num_states # initial state probabilities
    
        rho_init = 1e-2
        self.rho = rho_init * np.ones((num_states, num_states, num_obs, num_states)) # sufficient statistics; denoted 'phi' in Mongillo & Deneve
    
        self.g = np.multiply.outer(np.eye(num_states), np.eye(num_states))
        self.g = np.swapaxes(self.g, 1, 2) # g[i,j,l,h] = 1(i=l)1(j=h) as below Eq. (2.9) in Mongillo & Deneve

    def update(self, y, t, learn_theta=True):

        iid = self.iid

        Z, X = self.num_states, self.num_obs
        eta = discount(t)
    
        gamma = self.phi * np.expand_dims(self.theta[:,y], 0) / np.reshape(np.einsum('mn,n,m', self.phi, self.theta[:,y], self.probs_z), (1,1))
        # note that in the iid case, gamma has no dependence on the first axis as long as self.phi did not

        delta = np.zeros(self.num_obs)
        delta[y] = 1
        delta = np.reshape(delta, (1,1,X,1,1)) # indices = (i,j,k,l,h) in Eq. (2.11) of Mongillo & Deneve
        drho = np.multiply(delta, np.expand_dims(self.g, 2)) # insert k dimension between (i,j) and (l,h) dimensions of g
        drho = np.multiply(drho, np.reshape(self.probs_z, (1,1,1,Z,1))) # add dimensions to q_l for the preceding (i,j,k) and final (h) dimensions
        rho_previous = np.repeat(np.expand_dims(self.rho, -1), Z, -1) # add the 'h' index and just repeat, since the first term in Eq. (2.11) is h-independent
        self.rho = np.einsum('lh,ijklh->ijkh', gamma, (1 - eta) * rho_previous + eta * drho) 

        if not iid:
            self.probs_z = np.einsum('ml,m->l', gamma, self.probs_z)
        else:
            self.probs_z = np.einsum('ml,m->l', gamma, self.phi.mean(0))

        if t>1:
            self.phi = np.sum(self.rho,(2,3)) / np.expand_dims(np.sum(self.rho,(1,2,3)), -1)
            if iid:
                self.phi = np.repeat(self.phi.mean(0, keepdims=True), Z, 0) # average to effectively reduce from transition matrix to prior over z
            if learn_theta:
                self.theta = np.sum(self.rho,(0,3)) / np.expand_dims(np.sum(self.rho,(0,2,3)), -1)
            if np.sum(self.theta[:,y])==0: assert False

        if np.any(np.isnan(self.rho)):
            print('NaNs in online EM model sufficient statistics!!')
            exit()


class GaussianHMMonline:

    def __init__(self, num_states, probs_z=None, iid=False):
        ' implementation of the online EM algorithm in Cappe (2011), "Online EM Algorithm for HMMs" '

        self.num_states = num_states

        if iid: raise NotImplementedError

        self.t_min = 1

        if probs_z is not None:
            self.probs_z = probs_z
        else:
            self.probs_z = np.ones(num_states,)/num_states # np.array([0.2,0.8])

        # sufficient statistics
        rho_init = 1 ## making this close to zero or very large does not help
        self.rho_phi = rho_init * np.ones((num_states, num_states, num_states)) 
        self.rho_theta = rho_init * np.ones((num_states, num_states, 3)) # last dimension = sufficient statistics 's'

        prob_change = 0.55
        self.phi = (1 - prob_change) * np.eye(num_states) + prob_change * (1 - np.eye(num_states)) / (num_states - 1) # initial transition matrix estimate

        stdev_init = num_states
        self.mu, self.Sigma = np.arange(num_states), stdev_init * np.ones(num_states) # initial estimates of emission distributions

        self.theta = np.concatenate((np.expand_dims(self.mu, axis=0), np.expand_dims(self.Sigma, axis=0)))

    def g(self, y):

        return norm.pdf(y, self.mu, np.sqrt(self.Sigma))

    def update(self, y, t):

        Z = self.num_states

        gamma = discount(t)

        s = np.array([1, y, y*y]) # sufficient statistics for the Gaussian case

        r = np.expand_dims(self.probs_z, 1) * self.phi
        r /= np.expand_dims(np.einsum('i,ij->j', self.probs_z, self.phi), 0) # Eq. (23) in Cappe (2011) denotes phi as 'q' and probs_z as 'phi'

        self.probs_z = np.einsum('i,ij,j->j', self.probs_z, self.phi, self.g(y)) # Eq. (16)
        self.probs_z /= self.probs_z.sum()

        self.rho_phi = (1 - gamma) * np.einsum('ijl,lk->ijk', self.rho_phi, r) # Eq. (21)
        self.rho_phi += gamma * np.expand_dims(r, -1) * np.expand_dims(np.eye(Z), 0)
        self.rho_theta = (1 - gamma) * np.einsum('ils,lk->iks', self.rho_theta, r) # Eq. (22)
        self.rho_theta += gamma * np.reshape(s, (1,1,len(s))) * np.expand_dims(np.eye(Z), -1)

        if t>self.t_min:

            self.phi = np.einsum('ijk,k->ij', self.rho_phi, self.probs_z)
            self.phi /= self.phi.sum(1, keepdims=True)

            S_theta = np.einsum('iks,k->is', self.rho_theta, self.probs_z)
            self.mu = S_theta[:,1] / S_theta[:,0] # Eq. (25)
            self.Sigma = S_theta[:,2] / S_theta[:,0] - self.mu * self.mu # Eq. (26)

            self.theta = np.concatenate((np.expand_dims(self.mu, axis=0), np.expand_dims(self.Sigma, axis=0)))
