import numpy as np
import math
import itertools
from env import DynamicConfounderBanditGaussian, DynamicConfounderBanditDiscrete
from hmm import MultinomialHMMonline, GaussianHMMonline

from cmd_args import cmd_args


#prob_min = 1e-2 # minimum probability to assign to any latent state

lambda_mu = 1
var_r = 1

alpha_ucb = 3

def run(env, ep_length=int(1e5), num_episodes=1, oracle_posterior=False, offline_samples=True, Z=None, iid=False, algo='ts', reward_update=True):

    if oracle_posterior: assert Z is None # require access to the true number of latent states Z when oracle_posterior=True

    if Z is None: # use the true Z unless supplied with a model estimate for Z
        Z = env.Z
    K = env.K
    if isinstance(env, DynamicConfounderBanditDiscrete): X = env.X

    phi_star = env.latent_transition
    
    def episode(ep_length, filename):

        # prior over initial latent state
        probs_z_star = env.probs_latent_init
        if isinstance(env, DynamicConfounderBanditDiscrete):
            probs_z = np.random.dirichlet([1]*Z)
        else:
            probs_z = np.ones(Z,)/Z

        if isinstance(env, DynamicConfounderBanditDiscrete):
            hmm_model = MultinomialHMMonline(Z, X, probs_z, iid=iid)
            mu_star = env.probs_reward
        elif isinstance(env, DynamicConfounderBanditGaussian):
            hmm_model = GaussianHMMonline(Z, probs_z, iid=iid)
            mu_star = np.transpose(env.R_cond[:,:,0])

        # initial estimates for transition matrix and context distributions
        # phi, theta = hmm_model.phi, hmm_model.theta
        if offline_samples:
            assert isinstance(env, DynamicConfounderBanditDiscrete)
            num_samples = 5
            likelihood_x = np.zeros((env.X, Z))
            for z in range(Z):
                probs_latent = np.zeros(Z)
                probs_latent[z] += 1
                x_samples = env.sample_static(probs_latent, num_samples)[0]
                for x in range(env.X):
                    counts_x = np.sum(((x_samples==x)*1))
                    likelihood_x[x,z] = counts_x/num_samples
            likelihood_x = (1 - 1/num_samples) * likelihood_x + 1/(env.X*num_samples) # assign nonzero probability to x's that may not have been sampled
            hmm_model.theta = np.transpose(likelihood_x)

        # algorithm arrays:
        mu = np.zeros((K, Z))
        f_mu = np.zeros((K, Z))
        cov_inv_mu = lambda_mu * np.repeat(np.expand_dims(np.eye(Z), 0), K, axis=0)
        if algo is 'ts': mu_sample = np.empty((K, Z))
        if algo is 'ucb': action_values = np.empty(K)

        rewards_cum, rewards_cum_optimal = 0, 0
        rewards_cum_list, rewards_cum_optimal_list = [], []

        for t in range(ep_length):

            x = env.step()

            if oracle_posterior:
                hmm_model.probs_z = probs_z_star
            elif reward_update:
                hmm_model.probs_z = probs_z # update to include most recent reward

            if not oracle_posterior:
                # update HMM parameter estimates
                hmm_model.update(x, t+1)
                # update model posterior with context data
                probs_z = hmm_model.probs_z # hmm_model.update() above updated q

            # update oracle posterior with context data
            nll_x = -env.log_likelihood_x(x)
            probs_z_star = phi_star.dot(probs_z_star)
            probs_z_star *= np.exp(-nll_x)
            probs_z_star /= np.sum(probs_z_star, keepdims=True)

            if oracle_posterior: probs_z = probs_z_star

            # action selection
            for a in range(K):
                cov_a = np.linalg.inv(cov_inv_mu[a])
                if algo is 'ts':
                    mu_sample[a] = np.random.multivariate_normal(mu[a], cov = var_r * cov_a)
                elif algo is 'ucb':
                    action_values[a] = probs_z.dot(cov_a.dot(f_mu[a])) + alpha_ucb * np.sqrt(probs_z.dot(cov_a.dot(probs_z)))
            if algo is 'ts':
                a = np.argmax(mu_sample.dot(probs_z))
            elif algo is 'ucb':
                a = np.argmax(action_values)

            # optimal action
            a_star = np.argmax(mu_star.dot(probs_z_star))
        
            rewards_cum += mu_star[a].dot(probs_z_star) # the expected reward from ground truth parameters
            rewards_cum_list += [rewards_cum]
            rewards_cum_optimal += mu_star[a_star].dot(probs_z_star)
            rewards_cum_optimal_list += [rewards_cum_optimal]
        
            r = env.reward(a)

            # model update with reward data: update mean reward estimator and covariance
            cov_inv_mu[a] += np.outer(probs_z, probs_z)
            f_mu[a] += probs_z*r
            cov_a = np.linalg.inv(cov_inv_mu[a])
            mu[a] = cov_a.dot(f_mu[a])

            # update model posterior with reward data
            if isinstance(env, DynamicConfounderBanditDiscrete):
                mu = np.minimum(np.maximum(mu,0),1) ## this is a hack to keep Bernoulli probs between 0 and 1
                assert r==1 or r==0
                probs_z *= mu[a]*(r==1) + (1 - mu[a])*(r==0)
            elif isinstance(env, DynamicConfounderBanditGaussian):
                nll_r = (r - mu[a])**2 / (2*var_r)
                probs_z *= np.exp(-nll_r)
            probs_z /= np.sum(probs_z, keepdims=True)

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
        R = episode(ep_length, cmd_args.save_dir + '/ll' + algo + '_episode' + str(n))
        print('LL' + algo + ' REGRET FOR EPISODE:')
        print(R)
        print('\n')

