import numpy as np
#import random
import math
import copy
from env import DynamicConfounderBanditGaussian, DynamicConfounderBanditDiscrete
import env_params
import env_params_mining
from algorithm import run
from linear_bandit import run_linear
from cmd_args import cmd_args


#random.seed(0)
np.random.seed(0)


env_type = 'gaussian'
#env_type = 'discrete'

if env_type=='gaussian':

    # mining application parameters
    timescale = 'fast'
    theta_star, phi_star, reward_params = env_params_mining.inputs_mining_exp(timescale)

    Z = phi_star.shape[0]
    probs_latent_init = np.random.rand(Z,) # np.ones(Z,)/Z
    probs_latent_init /= probs_latent_init.sum()

    env = DynamicConfounderBanditGaussian(theta_star, probs_latent_init, phi_star, reward_params)

if env_type=='discrete':

    phi_star, probs_context, probs_reward, probs_latent_init = env_params.discrete_1()
    #phi_star, probs_context, probs_reward, probs_latent_init = env_params.discrete_2()
    X, Z = probs_context.shape
    K, _ = probs_reward.shape
    R = 2

    env = DynamicConfounderBanditDiscrete(X, K, Z, R, probs_context, probs_reward, probs_latent_init, phi_star) ## prob_change=prob_change

ep_length = 10000
num_episodes = 10

#run_linear(env, ep_length, num_episodes)

bandit_algo = 'ts'
#bandit_algo = 'ucb'
oracle_posterior=False
#oracle_posterior=True
offline_samples=False
run(env, ep_length, num_episodes, oracle_posterior=oracle_posterior, offline_samples=offline_samples, algo=bandit_algo)

