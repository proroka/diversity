# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:34:19 2016
@author: amandaprorok

"""


import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import pickle

# my modules
sys.path.append('../plotting')
sys.path.append('../utilities')
sys.path.append('..')

import funcdef_draw_network as nxmod

# -----------------------------------------------------------------------------#
# import data

load_data = True
load_run = 'T1'
load_prefix = "../data/" + load_run + "_"

# species-trait matrix
species_traits = pickle.load(open(load_prefix+"species_traits.p", "rb"))

# initial robot distribution
deploy_robots_init = pickle.load(open(load_prefix+"deploy_robots_init.p", "rb"))

# transition rates
transition_m_init = pickle.load(open(load_prefix+"transition_m_init.p", "rb"))

# graph
graph = pickle.load(open(load_prefix+"graph.p", "rb"))


# -----------------------------------------------------------------------------#
# setup

# constants
velocity_on_circle = 0.06
avoidance_velocity = 0.002
avoidance_range = -10 # no avoidance
min_velocity = 0.03
max_velocity = 0.08
task_radius = 0.05
arena_size = 3

num_robots = int(np.ceil(np.sum(deploy_robots_init)))
robots_pos_init = np.random.rand(num_robots,2)

# place robots in arena


# assign init tasks to all robots


# setup task sites


# get transition probabilities (from rate matrix)


# -----------------------------------------------------------------------------#
# main loop



# -----------------------------------------------------------------------------#
# save trajectories and species information


