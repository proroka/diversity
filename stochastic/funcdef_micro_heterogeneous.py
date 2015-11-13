# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:09:42 2015

@author: amanda
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx

from optimize_transition_matrix_hetero import *



# -----------------------------------------------------------------------------#
# initialize robots
# converts robot distribution to robot array (species and node specified for each robot)

def initialize_robots(deploy_robots_init):
    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]

    total_num_robots = np.sum(np.sum(deploy_robots_init))
    robots = np.zeros((total_num_robots,2))
    robots_per_species = np.sum(deploy_robots_init,0)
    ind = 0
    # allocate species in column 0, nodes in column 1
    for s in range(num_species):
        robots[ind:ind+robots_per_species[s],0] = s
        ind_2 = ind
        for n in range(num_nodes):
            v = deploy_robots_init[n,s]
            robots[ind_2:ind_2+v,1] = n
            ind_2 += v
        ind += robots_per_species[s]

    return robots



# -----------------------------------------------------------------------------#
# helper function that picks a random index given probabilities (summing to 1)

def pick_transition(p):
    rand = np.random.rand(1)
    v = 0
    for i in range(np.size(p)):
        v += p[i]
        if rand <= v:
            return i
    # Should not happen (unless probabilities do not sum to 1 exactly).
    return np.size(p) - 1


# -----------------------------------------------------------------------------#
# microscopic model

def microscopic_sim(num_timesteps, delta_t, robots_init, deploy_robots_init, transition_m):

    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]

    deploy_robots = np.zeros((num_nodes, num_timesteps, num_species))
    deploy_robots[:,0,:] = deploy_robots_init

    robots = robots_init.copy()

    # Pre-compute transition probabilities.
    ps = []
    for s in range(num_species):
        ks = transition_m[:,:,s] # transition rates
        ps.append(sp.linalg.expm(delta_t*ks)) # transition probabilities

    for t in range(1, num_timesteps):
        # Propagate previous state.
        deploy_robots[:, t, :] = deploy_robots[:, t-1, :]

        for r in range(robots.shape[0]):
            r_s = robots[r,0] # robot species
            r_n = robots[r,1] # current node
            node = pick_transition(ps[int(r_s)][:,r_n])
            # update
            deploy_robots[r_n, t, r_s] -= 1
            deploy_robots[node, t, r_s]  += 1
            robots[r,1] = node

    return (robots, deploy_robots)


# -----------------------------------------------------------------------------#
# distributed version of microscopic model
# transition_m: [num_nodes, num_nodes, num_species, num_nodes]


def microscopic_sim_distrib(num_timesteps, delta_t, robots_init, deploy_robots_init, transition_m):

    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]
    num_robots = robots_init.shape[0]

    deploy_robots = np.zeros((num_nodes, num_timesteps, num_species))
    deploy_robots[:,0,:] = deploy_robots_init

    robots = robots_init.copy()

    # Pre-compute transition probabilities. 4th dimension: distributed version, p's according to local-node belief
    ps = np.zeros((num_nodes, num_nodes, num_species, num_nodes))
    for nd in range(num_nodes):    
        for s in range(num_species):
            ks = transition_m[:,:,s,nd] # transition rates
            ps[:,:,s,nd] = sp.linalg.expm(delta_t*ks) # transition probabilities

    for t in range(1, num_timesteps):
        # Propagate previous state.
        deploy_robots[:, t, :] = deploy_robots[:, t-1, :]

        for r in range(num_robots):
            r_s = robots[r,0] # robot species
            r_n = robots[r,1] # current node
            node = pick_transition(ps[:,r_n,int(r_s),r_n])
            # update
            deploy_robots[r_n, t, r_s] -= 1
            deploy_robots[node, t, r_s]  += 1
            robots[r,1] = node

    return (robots, deploy_robots)

# -----------------------------------------------------------------------------#
    
def microscopic_sim_get_time(num_timesteps, delta_t, robots_init, deploy_robots_init, transition_m,
                             deploy_traits_desired, deploy_robots_desired, transform, match, min_val, slack):

    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]

    deploy_robots = np.zeros((num_nodes, num_timesteps, num_species))
    deploy_robots[:,0,:] = deploy_robots_init

    robots = robots_init.copy()

    # Pre-compute transition probabilities.
    ps = []
    for s in range(num_species):
        ks = transition_m[:,:,s] # transition rates
        ps.append(sp.linalg.expm(delta_t*ks)) # transition probabilities

    total_num_traits = np.sum(deploy_traits_desired)
    if deploy_robots_desired is not None:
        reached_robots = False
        total_num_robots = np.sum(deploy_robots_desired)
    for t in range(1, num_timesteps):
        # Check if we are done.
        if deploy_robots_desired is not None:
            diffr = np.abs(deploy_robots[:,t-1,:] - deploy_robots_desired)
            ratior = np.sum(diffr) / total_num_robots
            if ratior <= (min_val*slack):
                reached_robots = True
        traits = np.dot(deploy_robots[:,t-1,:], transform)
        if match == 0:
            diff = np.abs(np.minimum(traits - deploy_traits_desired, 0))
        else:
            diff = np.abs(traits - deploy_traits_desired)
        ratio = np.sum(diff) / total_num_traits
        if ratio <= min_val and reached_robots:
            return t - 1

        # Propagate previous state.
        deploy_robots[:, t, :] = deploy_robots[:, t-1, :]

        for r in range(robots.shape[0]):
            r_s = robots[r,0] # robot species
            r_n = robots[r,1] # current node
            node = pick_transition(ps[int(r_s)][:,r_n])
            # update
            deploy_robots[r_n, t, r_s] -= 1
            deploy_robots[node, t, r_s]  += 1
            robots[r,1] = node

    return num_timesteps - 1
    
    
    
