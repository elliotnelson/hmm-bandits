import numpy as np
from scipy.stats import norm
import math
import sys


def categorical(probs):
    u = np.random.uniform(size=probs.shape[0])
    u = np.expand_dims(u, axis=-1)
    u = np.repeat(u, probs.shape[-1], axis=-1)
    return (probs.cumsum(-1) >= u).argmax(-1)

class DynamicConfounderBanditDiscrete():

    def __init__(self, X, K, Z, R, probs_context, probs_reward, probs_latent_init, phi_star):

        assert R==2 ## methods below assume binary rewards

        self.X, self.K, self.Z = X, K, Z

        self.probs_context = probs_context
        self.probs_reward = probs_reward
        self.probs_latent_init = probs_latent_init

        self.latent_transition = phi_star
        ## self.latent_transition = (1 - prob_change*(1 + 1/(Z-1))) * np.eye(Z) + np.full((Z,Z), prob_change/(Z-1))
        ## self.latent_transition /= np.sum(self.latent_transition,1,keepdims=True)
        self.z = np.random.choice(np.arange(self.Z), p=probs_latent_init)

    def step(self):

        self.z = np.random.choice(np.arange(self.Z), p=self.latent_transition[:,self.z]) # state transition

        x = np.random.choice(np.arange(self.X), p=self.probs_context[:,self.z])
        return x

    def reward(self, action):
        r = np.random.uniform() < self.probs_reward[action,self.z]
        return r*1

    def sample_static(self, probs_latent, num_samples=1):

        z_samples = np.random.choice(np.arange(self.Z), size=num_samples, p=probs_latent)
        x_samples = categorical(np.transpose(self.probs_context[:,z_samples]))
        r_vec_samples = np.random.uniform(size=(self.K, num_samples)) < self.probs_reward[:,z_samples]
        a_opt_samples = np.argmax(r_vec_samples, 0)

        return (x_samples, r_vec_samples.transpose(), a_opt_samples)

    def log_likelihood_x(self, x):

        probs_z = np.maximum(self.probs_context[x], 1e-4)
        return np.log(probs_z)

    def log_likelihood_r(self, reward, action):

        ll = reward * np.log(self.probs_reward[action]) + (1 - reward) * np.log(1 - self.probs_reward[action])
        return ll

    def entropies_x(self):

        H = -np.einsum('xz,xz->z', self.probs_context, np.log(self.probs_context))
        return H

    def kldivs_x(self):
        'matrix of kldivs between z-conditional context likelihoods; element (i,j) = KL[p_i(x),p_j(x)]'
        'this method is not currently used'

        p = np.repeat(np.expand_dims(self.probs_context, -1), self.Z, axis=-1)
        q = np.repeat(np.expand_dims(self.probs_context, 1), self.Z, axis=1)
        return np.einsum('xij,xij->ij', p, np.log(p) - np.log(q))


class DynamicConfounderBanditGaussian():

    # this class uses Gaussian models for observations and rewards
    def __init__(self, X_cond_prob, Z_init_prior, latent_transition, R_cond):

        self.X_cond_prob, self.Z_init_prior, self.latent_transition = \
            X_cond_prob, Z_init_prior, latent_transition
        self.R_cond = R_cond

        self.Z, self.K = self.R_cond.shape[0], self.R_cond.shape[1]

        self.probs_latent_init = Z_init_prior
        self.z = np.random.choice(np.arange(self.Z), p=self.probs_latent_init)

    def step(self):

        self.z = np.random.choice(np.arange(self.Z), p=self.latent_transition[:,self.z]) # state transition

        x_mean = self.X_cond_prob[self.z, 0]
        x_stdev = self.X_cond_prob[self.z, 1]
        x = np.random.normal(x_mean, x_stdev)

        return x

    def sample_static(self, probs_latent, num_samples=1):

        z_samples = np.random.choice(np.arange(self.Z), size=num_samples, p=probs_latent)
        x_mean_samples = self.X_cond_prob[z_samples, 0]
        x_stdev_samples = self.X_cond_prob[z_samples, 1]
        r_mean_vec_samples = self.R_cond[z_samples, :, 0]
        r_stdev_vec_samples = self.R_cond[z_samples, :, 1]
        x_samples = np.random.normal(x_mean_samples, x_stdev_samples)
        r_vec_samples = np.random.normal(r_mean_vec_samples, r_stdev_vec_samples)
        #a_opt_samples = np.argmax(self.R_cond[z_samples,:,0], -1)
        a_opt_samples = np.argmax(r_vec_samples, 1)

        return (x_samples.reshape(-1,1), r_vec_samples.reshape(-1,1), a_opt_samples.reshape(-1,1)) 

    def reward(self, action):

        r_mean = self.R_cond[self.z, action, 0]
        r_stdev = self.R_cond[self.z, action, 1]
        r = np.random.normal(r_mean, r_stdev)

        return r

    def log_likelihood_x(self, x):

        x_means = self.X_cond_prob[:, 0]
        x_stdevs = self.X_cond_prob[:, 1]

        return norm.logpdf(x, x_means, x_stdevs)

    def log_likelihood_r(self, reward, action):

        r_means = self.R_cond[:, action, 0]
        r_stdevs = self.R_cond[:, action, 1]

        return norm.logpdf(reward, r_means, r_stdevs)

    def entropies_x(self):
        'vector of entropies of z-conditional context likelihoods'
        'this method is not currently used'

        x_means = self.X_cond_prob[:, 0]
        x_stdevs = self.X_cond_prob[:, 1]
        H = np.log(x_stdevs) + 0.5 * (np.log(2*math.pi) + 1)
        return H

    def kldivs_x(self):
        'matrix of kldivs between z-conditional context likelihoods; element (i,j) = KL[p_i(x),p_j(x)]'
        'this method is not currently used'

        x_means = self.X_cond_prob[:, 0]
        x_stdevs = self.X_cond_prob[:, 1]
        kldivs = np.repeat(np.expand_dims(x_means, 0), self.Z, axis=0) 
        kldivs -= np.repeat(np.expand_dims(x_means, 1), self.Z, axis=1)
        kldivs *= kldivs
        sigma1 = np.repeat(np.expand_dims(x_stdevs, 0), self.Z, axis=0)
        sigma2 = np.repeat(np.expand_dims(x_stdevs, 1), self.Z, axis=1)
        kldivs += np.square(sigma2)
        kldivs /= 2*np.square(sigma1)
        kldivs += np.log(sigma1) - np.log(sigma2) - 0.5
        return kldivs

