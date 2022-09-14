import numpy as np
import math
import itertools
from env import DynamicConfounderBanditGaussian, DynamicConfounderBanditDiscrete
from hmm import MultinomialHMMonline, GaussianHMMonline

from cmd_args import cmd_args


lambda_mu = 1
var_r = 1

def run_linear(env, ep_length=int(1e5), num_episodes=1, algo='TS'):

    assert isinstance(env, DynamicConfounderBanditGaussian)

    K = env.K
    mu_star = np.transpose(env.R_cond[:,:,0])

    phi_star = env.latent_transition

    def episode(ep_length, filename):

        probs_z_star = env.probs_latent_init

        ctx_dim = 1

        # algorithm arrays:
        mu = np.zeros((K, ctx_dim))
        f_mu = np.zeros((K, ctx_dim))
        cov_inv_mu = lambda_mu * np.repeat(np.expand_dims(np.eye(ctx_dim), 0), K, axis=0)
        if algo is 'TS': mu_sample = np.empty((K, ctx_dim))
        else: raise NotImplementedError

        rewards_cum, rewards_cum_optimal = 0, 0
        rewards_cum_list, rewards_cum_optimal_list = [], []

        for t in range(ep_length):

            x = env.step()

            # update oracle posterior with context data
            nll_x = -env.log_likelihood_x(x)
            probs_z_star = phi_star.dot(probs_z_star)
            probs_z_star *= np.exp(-nll_x)
            probs_z_star /= np.sum(probs_z_star, keepdims=True)

            # action selection
            for a in range(K):
                covar = np.linalg.inv(cov_inv_mu[a])
                if algo is 'TS':
                    mu_sample[a] = np.random.multivariate_normal(mu[a], cov = var_r * covar)
            if algo is 'TS':
                a = np.argmax(mu_sample.dot(x)) # when ctx_dim=1, this is just argmax(mu_sample)
            else: raise NotImplementedError

            # optimal action
            a_star = np.argmax(mu_star.dot(probs_z_star))
        
            rewards_cum += mu_star[a].dot(probs_z_star) # the expected reward from ground truth parameters
            rewards_cum_list += [rewards_cum]
            rewards_cum_optimal += mu_star[a_star].dot(probs_z_star)
            rewards_cum_optimal_list += [rewards_cum_optimal]
        
            r = env.reward(a)

            # model update with reward data: update mean reward estimator and covariance
            cov_inv_mu[a] += np.outer(x, x)
            f_mu[a] += x*r
            cov_a = np.linalg.inv(cov_inv_mu[a])
            mu[a] = cov_a.dot(f_mu[a])

            r_star = env.reward(a_star)
            nll_r_star = -env.log_likelihood_r(r, a_star)
            # update oracle posterior with reward data
            probs_z_star *= np.exp(-nll_r_star)
            probs_z_star /= np.sum(probs_z_star, keepdims=True)

        np.save(filename + '_rewards', np.array(rewards_cum_list))
        np.save(filename + '_regret', np.array(rewards_cum_optimal_list) - np.array(rewards_cum_list))
    
        return rewards_cum_optimal_list[-1] - rewards_cum_list[-1]

    for n in range(num_episodes):
        print('STARTING EPISODE %d' % n)
        print('\n')
        R = episode(ep_length, cmd_args.save_dir + '/linTS_episode' + str(n))
        print('REGRET FOR EPISODE:')
        print(R)
        print('\n')

