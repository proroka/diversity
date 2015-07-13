# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:33:04 2015
@author: amanda

"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx

from optimize_transition_matrix_hetero_fast_gradient import *


# -----------------------------------------------------------------------------#
# sample a random transition matrix

def random_transition_matrix(num_nodes, max_rate):

    random_transition = np.random.rand(num_nodes, num_nodes) * max_rate
    # k_ii is -sum(k_ij) s.t. sum(column)=0; ensures constant total number of robots
    np.fill_diagonal(random_transition, np.zeros(num_nodes))
    np.fill_diagonal(random_transition, -np.sum(random_transition,0))

    return random_transition


# -----------------------------------------------------------------------------#
# run euler integration to sample random end state

def sample_final_robot_distribution(deploy_robots_init, random_transition, t_max, delta_t):

    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]

    #delta_t = 0.1
    num_iter = int(t_max / delta_t)
    deploy_robots_sample = np.zeros((num_nodes, num_iter, num_species))
    for s in range(num_species):
        deploy_robots_sample[:,0,s] = deploy_robots_init[:,s]
        for t in range(1,num_iter):
            deploy_robots_sample[:,t,s] = deploy_robots_sample[:,t-1,s] + delta_t*np.dot(random_transition, deploy_robots_sample[:,t-1,s])

    return deploy_robots_sample[:,t_max-1,:]



# -----------------------------------------------------------------------------#
# run euler integration and return time evolution

def run_euler_integration(deploy_robots_init, transition_m, t_max, delta_t):

    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]

    #delta_t = 0.01
    num_iter = int(t_max / delta_t)
    
    deploy_robots = np.zeros((num_nodes, num_iter, num_species))
    for s in range(num_species):
        deploy_robots[:,0,s] = deploy_robots_init[:,s]
        for t in range(1,num_iter):
            deploy_robots[:,t,s] = deploy_robots[:,t-1,s] + delta_t*np.dot(transition_m[:,:,s], deploy_robots[:,t-1,s])

    return deploy_robots


# -----------------------------------------------------------------------------#
# find optimal transition matrix

def optimal_transition_matrix(adjacency_m, deploy_robots_init, deploy_traits_desired, 
                              species_traits, max_time, max_rate, l_norm, match, optimizing_t, force_steady_state):

    find_optimal = True

    verbose = False
    if find_optimal:
        transition_m = Optimize_Hetero_Fast(adjacency_m, deploy_robots_init, deploy_traits_desired, 
                                            species_traits, max_time, max_rate, verbose, l_norm, match,
                                            optimizing_t, force_steady_state)
    else:
        # create random transition matrix
        num_nodes = adjacency_m.shape[0]
        num_species = species_traits.shape[0]
        transition_m = np.zeros((num_nodes,num_nodes,num_species))
        for i in range(num_species):
            transition_m[:,:,i] = np.random.rand(num_nodes, num_nodes) * adjacency_m
            # k_ii is -sum(k_ij) s.t. sum(column)=0; ensures constant total number of robots
            np.fill_diagonal(transition_m[:,:,i], -np.sum(transition_m[:,:,i],0))

    return transition_m






