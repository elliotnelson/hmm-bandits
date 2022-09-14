import numpy as np
import random


#random.seed(0)
np.random.seed(0)

def discrete_2():

    Z = 4
    prob_change = 0.25
    phi_star = (1 - prob_change) * np.eye(Z) + prob_change/(Z-1) * (1 - np.eye(Z))
    probs_latent_init = np.ones((Z,))/Z

    X = 12
    x_shift_per_z = int(X/Z)
    x_vals_per_z = x_shift_per_z
    x_probs = np.array([1/x_vals_per_z]*x_vals_per_z + [0]*(X-x_vals_per_z))
    theta_star = np.empty((X,Z))
    for z in range(Z):
        theta_star[:,z] = np.roll(x_probs, x_shift_per_z*z)

    K = 8
    mu_min, mu_max = 0, 1
    mu_star = np.random.uniform(mu_min, mu_max, (K, Z))

    return phi_star, theta_star, mu_star, probs_latent_init

def discrete_1():

    Z = 2
    prob_change = 0.1
    phi_star = (1 - prob_change) * np.eye(Z) + prob_change/(Z-1) * (1 - np.eye(Z)) # latent transition matrix
    
    R = 2

    probs_latent_init = np.array([0.62441598, 0.37558402]) # sampled randomly
    #probs_latent_init = np.ones(Z,)/Z
    #probs_latent_init = np.random.dirichlet([1]*Z) 

    K, X = 2, 4
    mu = 0.5
    Delta = 0.1
    px = 0.1
    probs_context = np.empty((X,Z))
    probs_context[:,0] = np.array([px]*int(X/2) + [1-px]*int(X/2))
    probs_context[:,1] = np.array([1-px]*int(X/2) + [px]*int(X/2))
    probs_context /= np.sum(probs_context,0,keepdims=True)
    #probs_reward = np.random.uniform(size=(K,Z))
    probs_reward = np.empty((K,Z))
    probs_reward[:,0] = np.tile(np.array([mu-Delta, mu+Delta]), int(K/2))
    probs_reward[:,1] = np.tile(np.array([mu+Delta, mu-Delta]), int(K/2))

    return phi_star, probs_context, probs_reward, probs_latent_init
