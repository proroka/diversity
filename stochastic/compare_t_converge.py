# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:40:07 2015
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
from optimize_transition_matrix_hetero import *
from optimize_transition_matrix_berman import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod


# -----------------------------------------------------------------------------#

# 1. create graph
# 2. initialize robot deployment on nodes
# 3. initialize transition matrix
# 4. run macro-discrete model  / micro simulator
# 5. plot evolution of robot population per node

# -----------------------------------------------------------------------------#
# initialize world and robot community

run = 'V13'

save_data = True
save_plots = True
fixed_species = False

tstart = time.strftime("%Y%m%d-%H%M%S")

# simulation parameters
t_max = 10.0 # influences desired state and optmization of transition matrix
t_max_sim = 10.0 # influences simulations and plotting
num_iter = 2 # iterations of micro sim
delta_t = 0.04 # time step
max_rate = 2.0 # Maximum rate possible for K.
num_graph_iter = 20

# cost function
l_norm = 2 # 2: quadratic 1: absolute
match = 1 # 1: exact 0: at-least



# -----------------------------------------------------------------------------#
# find time at which min ratio is found

num_tot_iter = num_iter * num_graph_iter

min_ratio = 0.05
t_min_mic = np.zeros((num_tot_iter))
t_min_mic_ber = np.zeros((num_tot_iter))
t_min_adp = np.zeros((num_tot_iter))
t_min_mac = np.zeros((num_graph_iter))
t_min_mac_ber = np.zeros((num_graph_iter))

rank_Q = np.zeros((num_graph_iter))

for g in range(num_graph_iter):

    ranks = np.array([3, 4])
    
    # -----------------------------------------------------------------------------#
    # initialize robots
    num_nodes = 8
    # set of traits
    num_traits = 6
    max_trait_values = 2 # [0,1]: trait availability
    # robot species
    num_species = 6
    max_robots = 200 # maximum number of robots per node
    deploy_robots_init = np.random.randint(0, max_robots, size=(num_nodes, num_species))
    # ensure each species has at least 1 trait, and that all traits are present
    
    if(num_species==4 and num_traits==4 and fixed_species):
        rank = ranks[np.mod(g,2)]
        species_traits = get_species_trait_matrix_44(rank)    
    else:
        species_traits = np.zeros((num_species, num_traits))
        while ((min(np.sum(species_traits,0))==0 or min(np.sum(species_traits,1))==0)):
            species_traits = np.random.randint(0, max_trait_values, (num_species, num_traits))
       
    rank_Q[g] = np.linalg.matrix_rank(species_traits)
    # generate a random end state
    random_transition = random_transition_matrix(num_nodes, max_rate/2)  # Divide max_rate by 2 for the random matrix to give some slack.        
    # sample final desired trait distribution based on random transition matrix   
    deploy_robots_final = sample_final_robot_distribution(deploy_robots_init, random_transition, t_max*4., delta_t)
    deploy_traits_init = np.dot(deploy_robots_init, species_traits)
    deploy_traits_desired = np.dot(deploy_robots_final, species_traits)
    # if 'at-least' cost function, reduce number of traits desired
    if match==0:
        deploy_traits_desired *= (np.random.rand()*0.1 + 0.85)
        print "total traits, at least: \t", np.sum(np.sum(deploy_traits_desired))

    # initialize robots
    robots = initialize_robots(deploy_robots_init)
        
    # -----------------------------------------------------------------------------#
    # initialize graph: all species move on same graph
    graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.6)
    
    # get the adjencency matrix
    adjacency_m = nx.to_numpy_matrix(graph)
    adjacency_m = np.squeeze(np.asarray(adjacency_m))
    
    # -----------------------------------------------------------------------------#
    # find optimal transition matrix for plain micro
    
    init_transition_values = np.array([])
    transition_m_init = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init, deploy_traits_desired,
                                                  species_traits, t_max, max_rate,l_norm, match, optimizing_t=True, force_steady_state=2.0)
    
    # -----------------------------------------------------------------------------#
    # Berman's method 
    transition_m_berman = Optimize_Berman(adjacency_m, deploy_robots_init, deploy_robots_final, species_traits,max_rate, max_time=t_max, verbose=True)
    
    
    # -----------------------------------------------------------------------------#
    # run microscopic stochastic simulation
    
    num_timesteps = int(t_max_sim / delta_t)
    
    deploy_robots_micro = np.zeros((num_nodes, num_timesteps, num_species, num_iter))
    deploy_robots_mic_ber = np.zeros((num_nodes, num_timesteps, num_species, num_iter))
    for it in range(num_iter):
        print "Iteration: ", it
        robots_new, deploy_robots_micro[:,:,:,it] = microscopic_sim(num_timesteps, delta_t, robots, deploy_robots_init, transition_m_init)
        robots_new_ber, deploy_robots_mic_ber[:,:,:,it] = microscopic_sim(num_timesteps, delta_t, robots, deploy_robots_init, transition_m_berman)
        tot_it = g*num_iter+it        
        t_min_mic[tot_it] = get_traits_ratio_time(deploy_robots_micro[:,:,:,it], deploy_traits_desired, species_traits, match, min_ratio)
        t_min_mic_ber[tot_it] = get_traits_ratio_time(deploy_robots_mic_ber[:,:,:,it], deploy_traits_desired, species_traits, match, min_ratio)

    # -----------------------------------------------------------------------------#
    # run adaptive microscopic stochastic simulation, RHC
    
    numts_window = 2 # number of time steps per window
    t_window = float(numts_window) * delta_t
    slices = int(t_max_sim / t_window)
    
    deploy_robots_micro_adapt = np.zeros((num_nodes, num_timesteps, num_species, num_iter))
    
    init_transition_values = np.array([])
    for it in range(num_iter):
    
        transition_m = transition_m_init.copy()
        deploy_robots_init_slice = deploy_robots_init.copy()
        robots_slice = robots.copy()
        for sl in range(slices):
            print "G iter: ", g+1, "/", num_graph_iter, "Adp iter: ", it+1 , "/", num_iter, "Slice: ", sl+1,"/", slices
            robots_slice, deploy_robots_micro_slice = microscopic_sim(numts_window + 1, delta_t, robots_slice, deploy_robots_init_slice, transition_m)
            deploy_robots_init_slice = deploy_robots_micro_slice[:,-1,:]
            # put together slices
            deploy_robots_micro_adapt[:,sl*numts_window:(sl+1)*numts_window,:,it] = deploy_robots_micro_slice[:,:-1,:]
            # calculate adapted transition matrix
            init_transition_values = transition_m.copy()
            transition_m = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_slice, deploy_traits_desired,
                                                     species_traits, t_max, max_rate,l_norm, match, optimizing_t=True,
                                                     force_steady_state=0.0)
            #print transition_m.reshape((num_nodes, num_nodes))
        # Simulate the last extra bit.
        timesteps_left = num_timesteps - slices * numts_window
        assert timesteps_left >= 0
        if timesteps_left > 0:
            robots_slice, deploy_robots_micro_slice = microscopic_sim(timesteps_left + 1, delta_t, robots_slice, deploy_robots_init_slice, transition_m)
            deploy_robots_micro_adapt[:,-timesteps_left:,:,it] = deploy_robots_micro_slice[:,:-1,:]
        
        tot_it = g*num_iter+it    
        t_min_adp[tot_it] = get_traits_ratio_time(deploy_robots_micro_adapt[:,:,:,it], deploy_traits_desired, species_traits, match, min_ratio)    

    # -----------------------------------------------------------------------------#
    # euler integration
    
    deploy_robots_mac_ber = run_euler_integration(deploy_robots_init, transition_m_berman, t_max_sim, delta_t)
    deploy_robots_euler = run_euler_integration(deploy_robots_init, transition_m_init, t_max_sim, delta_t)
    
    t_min_mac_ber[g] = get_traits_ratio_time(deploy_robots_mac_ber, deploy_traits_desired, species_traits, match, min_ratio)
    t_min_mac[g] = get_traits_ratio_time(deploy_robots_euler, deploy_traits_desired, species_traits, match, min_ratio)

# -----------------------------------------------------------------------------#
# save variables

if save_data:
    
    tend = time.strftime("%Y%m%d-%H%M%S")

    prefix = "./data/" + run + "_"
    print "Time start: ", tstart
    print "Time end: ", tend
    
    pickle.dump(species_traits, open(prefix+"species_traits.p", "wb"))
    pickle.dump(graph, open(prefix+"graph.p", "wb"))
    pickle.dump(deploy_robots_init, open(prefix+"deploy_robots_init.p", "wb"))
    pickle.dump(deploy_robots_final, open(prefix+"deploy_robots_final.p", "wb"))
    pickle.dump(deploy_traits_init, open(prefix+"deploy_traits_init.p", "wb"))
    pickle.dump(deploy_traits_desired, open(prefix+"deploy_traits_desired.p", "wb"))
    pickle.dump(deploy_robots_micro_adapt, open(prefix+"deploy_robots_micro_adapt.p", "wb"))
    pickle.dump(deploy_robots_micro, open(prefix+"deploy_robots_micro.p", "wb"))
    pickle.dump(deploy_robots_euler, open(prefix+"deploy_robots_euler.p", "wb"))
    pickle.dump(t_min_mic, open(prefix+"t_min_mic.p", "wb"))
    pickle.dump(t_min_mac, open(prefix+"t_min_mac.p", "wb"))
    pickle.dump(t_min_adp, open(prefix+"t_min_adp.p", "wb"))
    pickle.dump(t_min_mic_ber, open(prefix+"t_min_mic_ber.p", "wb"))
    pickle.dump(t_min_mac_ber, open(prefix+"t_min_mac_ber.p", "wb"))
    pickle.dump(rank_Q, open(prefix+"rank_Q.p", "wb"))

# -----------------------------------------------------------------------------#
# plots

# plot traits ratio
fig1 = plot_traits_ratio_time_micmicmac(deploy_robots_micro, deploy_robots_micro_adapt, deploy_robots_euler, deploy_traits_desired,species_traits, delta_t, match)

# plot time at which min ratio reached
fig2 = plot_t_converge(delta_t,t_min_mic, t_min_adp, t_min_mac, t_min_mic_ber)


# -----------------------------------------------------------------------------#
# save plots
 
if save_plots:
    prefix = "./plots/" + run + "_"                        
    fig1.savefig(prefix+'micmicmac.eps') 
    fig2.savefig(prefix+'time_converge.eps') 


