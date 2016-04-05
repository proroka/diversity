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
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod
from generate_Q import *

# -----------------------------------------------------------------------------#

# 1. setup: create graph
# 2. setup: initialize robot deployment on nodes
# 3. setup: initialize transition matrix
# 4. optimize transition rates
#
# -----------------------------------------------------------------------------#
# initialize world and robot community

run = 'RC02'

load_data = True
load_run = 'RC01'
load_prefix = "../data/RCx/" + load_run + '/' + load_run + "_"

run_optim = 1
run_micro = False

save_data = True

# random initial and final trait distributions
fix_init = True
fix_final = True

plot_graph = False
plot_run = 1

tstart = time.strftime("%Y%m%d-%H%M%S")
print str(run)
print "Time start: ", tstart

# simulation parameters
t_max = 8.0 # influences desired state and optmization of transition matrix
t_max_sim = 7.0 # influences simulations and plotting
delta_t = 0.04 # time step
max_rate = 1.0 # Maximum rate possible for K.

# graph
num_nodes = 6
half_num_nodes = num_nodes / 2    

# cost function
l_norm = 2 # 2: quadratic 1: absolute
match = 1 # 1: exact 0: at-least
match_margin = 0.2 # used when match=0 

# robot species
total_num_robots = 100.0 
num_species = 4
num_traits = 5
desired_rank = num_species

# privacy mechanism
# lap = 1.5
# careful: these parameters should never be == 0
range_lambda = [2.5, 4.0]
range_alpha = [1.0, 1.5, 2.0]
range_beta = [4.0, 5.0, 6.0]

num_sample_iter = 2


# -----------------------------------------------------------------------------#
# initialize distributions

    
# generate random robot species with fixed diversity
if not load_data:
    rk = 0
    print 'Computing Species-Trait matrix'
    while rk != desired_rank:
        species_traits, rk, s = generate_Q(num_species, num_traits)
else:
    species_traits = pickle.load(open(load_prefix+"species_traits.p", "rb"))
    print species_traits    

# generate a random end state
random_transition = random_transition_matrix(num_nodes, max_rate/2)  # Divide max_rate by 2 for the random matrix to give some slack.
deploy_robots_init = np.random.randint(0, 100, size=(num_nodes, num_species))
# normalize
deploy_robots_init = deploy_robots_init * total_num_robots / np.sum(np.sum(deploy_robots_init, axis=0))
sum_species = np.sum(deploy_robots_init,axis=0)


if fix_init:
    deploy_robots_init[half_num_nodes:,:] = 0
    # normalize
    deploy_robots_init = deploy_robots_init * total_num_robots / np.sum(np.sum(deploy_robots_init, axis=0))
    sum_species = np.sum(deploy_robots_init,axis=0)
if load_data:
    deploy_robots_init = pickle.load(open(load_prefix+"deploy_robots_init.p", "rb"))
    sum_species = np.sum(deploy_robots_init,axis=0)

print "total robots, init: \t", np.sum(np.sum(deploy_robots_init))

        
# sample final desired trait distribution based on random transition matrix
deploy_robots_final = sample_final_robot_distribution(deploy_robots_init, random_transition, t_max*4., delta_t)
if fix_final: 
    deploy_robots_final[:half_num_nodes,:] = 0
    ts = np.sum(deploy_robots_final, axis=0)
    for i in range(num_species):    
        deploy_robots_final[:,i] = deploy_robots_final[:,i] / float(ts[i]) * float(sum_species[i])
        
# trait distribution
if not load_data:
    deploy_traits_init = np.dot(deploy_robots_init, species_traits)
    deploy_traits_desired = np.dot(deploy_robots_final, species_traits)
else:
    deploy_traits_init = pickle.load(open(load_prefix+"deploy_traits_init.p", "rb"))
    deploy_traits_desired = pickle.load(open(load_prefix+"deploy_traits_desired.p", "rb"))
# if 'at-least' cost function, reduce number of traits desired
if match==0:
    deploy_traits_desired *= (np.random.rand()*match_margin + (1.0-match_margin))
    print "total traits, at least: \t", np.sum(np.sum(deploy_traits_desired))

# initialize robots
robots = initialize_robots(deploy_robots_init)

print "total robots, init: \t", np.sum(np.sum(deploy_robots_init))
print "total traits, init: \t", np.sum(np.sum(deploy_traits_init))


# -----------------------------------------------------------------------------#
# initialize graph

if not load_data:
    graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.7)
else:
    graph = pickle.load(open(load_prefix+"graph.p", "rb"))

# get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))

if plot_graph:
    plt.axis('equal')
    fig1 = nxmod.draw_circular(deploy_traits_init, graph,linewidths=3)
    plt.show()
    plt.axis('equal')
    fig2  = nxmod.draw_circular(deploy_traits_desired, graph, linewidths=3)
    plt.show()
    


# -----------------------------------------------------------------------------#
# optimization

# careful: these parameters should never be == 0

num_timesteps = int(t_max_sim / delta_t)

traj_ratio = {}
#traj_robots = {}

for el in range(len(range_lambda)):
    lap = range_lambda[el]
    for a in range(len(range_alpha)):
        alpha = range_alpha[a]
        for b in range(len(range_beta)):        
            beta = range_beta[b]
            
            ratio = np.zeros((num_timesteps, num_sample_iter))
            deploy_robots_euler_it = np.zeros((num_nodes, num_timesteps, num_species, num_sample_iter))
            for i in range(num_sample_iter):
            # optimize based on noisy initial state
                
                lap_val = np.random.laplace(loc=0.0, scale=lap, size=(num_nodes, num_species))
                deploy_robots_init_noisy = deploy_robots_init + lap_val
                
                # normalize robot distribution so that correct robots per species available
                ts = np.sum(deploy_robots_init_noisy, axis=0)
                for s in range(num_species):    
                    deploy_robots_init_noisy[:,s] = deploy_robots_init_noisy[:,s] / float(ts[s]) * float(sum_species[s])
            
                init_transition_values = np.array([])
                if run_optim:
                    print 'Optimizing rates...'
                    sys.stdout.flush()
                    transition_m_init = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_noisy, deploy_traits_desired,
                                                              species_traits, t_max, max_rate,l_norm, match, optimizing_t=True, force_steady_state=4.0, alpha=alpha, beta=beta)
                
                # run euler integration, evaluate trajectory based on true initial state with K from noisy optimization above
                deploy_robots_euler_it[:,:,:,i] = run_euler_integration(deploy_robots_init, transition_m_init, t_max_sim, delta_t)
                
                # store error ratio
                ratio[:,i] = get_traits_ratio_time_traj(deploy_robots_euler_it[:,:,:,i], deploy_traits_desired, species_traits, match)
            
            traj_ratio[(lap, alpha, beta)] = ratio.copy()
            #traj_robots[(lap, alpha, beta)] = deploy_robots_euler_it.copy()

# -----------------------------------------------------------------------------#
# plot simulations

if plot_run:
    i = 0
    fig = plot_traits_ratio_time_mac(deploy_robots_euler_it[:,:,:,i], deploy_traits_desired, species_traits, delta_t, match)
     
     

# -----------------------------------------------------------------------------#
# save variables

if save_data:
    
    tend = time.strftime("%Y%m%d-%H%M%S")
   
    prefix = "../data/RCx/" + run + "_"
    print str(run)    
    print "Time start: ", tstart
    print "Time end: ", tend
    
    pickle.dump(species_traits, open(prefix+"species_traits.p", "wb"))
    pickle.dump(graph, open(prefix+"graph.p", "wb"))
    pickle.dump(deploy_robots_init, open(prefix+"deploy_robots_init.p", "wb"))
    pickle.dump(deploy_traits_init, open(prefix+"deploy_traits_init.p", "wb"))
    pickle.dump(deploy_traits_desired, open(prefix+"deploy_traits_desired.p", "wb"))
        
    pickle.dump(max_rate, open(prefix+"max_rate.p", "wb"))
    pickle.dump(t_max_sim, open(prefix+"t_max_sim.p", "wb"))
    pickle.dump(num_timesteps, open(prefix+"num_timesteps.p", "wb"))

    pickle.dump(num_sample_iter, open(prefix+"num_sample_iter.p", "wb"))
    
    pickle.dump(range_alpha, open(prefix+"range_alpha.p", "wb"))
    pickle.dump(range_beta, open(prefix+"range_beta.p", "wb"))
    pickle.dump(range_lambda, open(prefix+"range_lambda.p", "wb"))
    
    pickle.dump(traj_ratio, open(prefix+"traj_ratio.p", "wb"))

   
    

