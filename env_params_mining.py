import numpy as np
import math


# function builds the inputs for the mining example (Gaussian version) of a non stationary latent bandit
# the actions are: heavy and light mining
def inputs_mining_exp(timescale):

  # parameters for P(X|Z)
  beta_0 = -0.18
  beta_1 = 1.32
  sigma = 0.62
  tau = 0.45

  # parameters for P(R|A,Z)
  # heavy vs. light: heavy has more reward fraction and more fixed cost but less variable cost
  reward_factor_per_grade_heavy = 0.4
  cost_heavy = 1
  var_cost_sigma_heavy = 0.5

  reward_factor_per_grade_light = 0.1
  cost_light = 0.5
  var_cost_sigma_light = 1

  Z_states = [1, 2, 3]  # 3 rock classes - higher is better, i.e. has higher oxide grade
  A_states = ['heavy', 'light']  # heavy mining vs. light mining actions

  # P(X|Z)
  mean_X_given_Z = [beta_0 + (beta_1 * Z) for Z in Z_states]
  stdev_X_given_Z = [math.sqrt((sigma**2) + (tau**2))] * len(Z_states)

  # P(R|Z,A)
  mean_R_given_Z_heavy = [(mu * reward_factor_per_grade_heavy) - cost_heavy for mu in mean_X_given_Z]
  stdev_R_given_Z_heavy = \
    [math.sqrt((reward_factor_per_grade_heavy * (sigma**2)) + (var_cost_sigma_heavy ** 2))] * len(Z_states)

  mean_R_given_Z_light = [(mu * reward_factor_per_grade_light) - cost_light for mu in mean_X_given_Z]
  stdev_R_given_Z_light = \
    [math.sqrt((reward_factor_per_grade_light * (sigma**2)) + (var_cost_sigma_light ** 2))] * len(Z_states)

  # latent transition matrix = P(Z'|Z)
  phi_star_fast = np.array([[0.7,0.25,0.05],[0.25,0.5,0.25],[0.05,0.25,0.7]])
  phi_star_slow = np.array([[0.98, 0.01, 0.01],[0.01,0.98,0.01],[0.01,0.01,0.98]])
  if timescale=='fast':
    phi_star = phi_star_fast
  elif timescale=='slow':
    phi_star = phi_star_slow

  Z, K = len(Z_states), len(A_states)

  # convert to numpy arrays
  theta_star = np.transpose(np.array([mean_X_given_Z, stdev_X_given_Z]))
  reward_params = np.empty((Z, K, 2))
  reward_params[:,0,0] = np.array(mean_R_given_Z_heavy)
  reward_params[:,1,0] = np.array(mean_R_given_Z_light)
  reward_params[:,0,1] = np.array(stdev_R_given_Z_heavy)
  reward_params[:,1,1] = np.array(stdev_R_given_Z_light)

  return theta_star, phi_star, reward_params

