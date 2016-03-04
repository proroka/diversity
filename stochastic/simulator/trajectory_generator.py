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
transition_r = pickle.load(open(load_prefix+"transition_m_init.p", "rb"))

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

dt = 2.
max_time = 10.
T = np.arange(0,max_time,dt)
num_timestep = np.size(T)
#dt_per_slot = round(time_per_slot / dt)
#dt_for_setup = round(setup_time / dt)

num_robots = int(np.ceil(np.sum(deploy_robots_init)))
num_species = np.size(species_traits,0)
num_traits = np.size(species_traits,1)
num_tasks = np.size(deploy_robots_init,0)
sum_species = np.sum(deploy_robots_init,0)
robots_pos_init = np.random.rand(num_robots,2)
robots_task_init = np.zeros((num_robots));


# assign init tasks to all robots
temp_i = 1
max_temp_i = 0
for si in range(num_species):
    max_temp_i = max_temp_i + sum_species[si];
    for ti in range(num_tasks):
        nb = int(round(deploy_robots_init[ti,si]));
        for nbi in range(nb):
            robots_task_init[temp_i-1] = ti;
            temp_i = temp_i + 1;
            # Avoid eventual rounding problems.
            if temp_i > max_temp_i:
                break
            
   
# initialize robots
robots_pos = np.zeros((num_robots, num_timesteps, 2))
robots_task = np.zeros((num_robots, num_timesteps, 1))
for i in range(num_robots):
    robots_pos[i, 0, :] = robots_pos_init[i, :]
    robots_task[i, 0] = robots_task_init[i]


# get transition probabilities (from rate matrix)
transition_p = np.zeros((num_tasks, num_tasks, num_species))
for i in range(num_species):
    transition_p[:,:,i] = sp.linalg.expm(transition_r[:,:,i] * dt)


# setup task sites
task_sites = np.zeros((num_tasks, 2));
if num_tasks == 1:
    # hard code 1st task site
    task_sites[0, :] = np.array([1.5, 1.5])
else:
    if num_tasks == 4:
        # % CW from top left: 1-3-4-2
        o = 0.65
        c = 1.5
        task_sites[1, :] = [c-o, c+o];
        task_site[2, :] = [c-o, c-o];
        task_sites[3, :] = [c+o, c+o];
        task_sites[4, :] = [c+o, c-o];
    else:
        for i in range(num_tasks):
            a = (i - 1.) / num_tasks * 2. * np.pi
            task_sites[i, :] = np.array([np.cos(a), np.sin(a)]) * (arena_size / 2.8 - task_radius - 0.1) + arena_size / 2;
        

# -----------------------------------------------------------------------------#
# main loop



# -----------------------------------------------------------------------------#
# save trajectories and species information


