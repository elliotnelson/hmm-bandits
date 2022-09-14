import argparse 
from argparse import Namespace
#import os
#import torch
#import numpy as np
#import logging


cmd_opt = argparse.ArgumentParser(description='Argparser for confounded multi-arm bandits', allow_abbrev=False)

cmd_opt.add_argument('-seed', type=int, default=0, help='new vs fixed random seed')
cmd_opt.add_argument('-num_ep', type=int, default=10, help='number of episodes to run')
cmd_opt.add_argument('-ep_length', type=int, default=int(1e4), help='episode length')
cmd_opt.add_argument('-save_dir', type=str, default='./scratch', help='save folder')
cmd_opt.add_argument('-load_dir', type=str, default='./scratch', help='load folder')

cmd_args = cmd_opt.parse_args()
