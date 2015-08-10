# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:11:09 2015
@author: amanda

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

run = 'V14'

save_data = True
save_plots = True
fixed_species = False

tstart = time.strftime("%Y%m%d-%H%M%S")

# simulation parameters
t_max = 10.0 # influences desired state and optmization of transition matrix
t_max_sim = 10.0 # influences simulations and plotting
num_iter = 1 # iterations of micro sim
delta_t = 0.04 # time step
max_rate = 2.0 # Maximum rate possible for K.
num_graph_iter = 1

# cost function
l_norm = 2 # 2: quadratic 1: absolute
match = 1 # 1: exact 0: at-least


# -----------------------------------------------------------------------------#
# initialize robots and graph

num_nodes = 8

# load Qs
num_species = 10
num_traits = 10

run = 'S10_U10'
prefix = "./data/Q/" + run + "_"

q_rs = pickle.load(open(prefix+"q_rs.p", "rb"))
q_rs1 = pickle.load(open(prefix+"q_rs1.p", "rb"))
q_rs2 = pickle.load(open(prefix+"q_rs2.p", "rb"))
q_rs_all = np.concatenate((q_rs, q_rs1, q_rs2),axis=2)

num_q_iter = q_rs_all.shape[2]
 
# -----------------------------------------------------------------------------#
# find time at which min ratio is found

num_mic_iter = num_iter * num_graph_iter * num_q_iter
num_mac_iter = num_graph_iter * num_q_iter

min_ratio = 0.05
t_min_mic = np.zeros((num_mic_iter))
t_min_mic_ber = np.zeros((num_mic_iter))
t_min_mac = np.zeros((num_mac_iter))
t_min_mac_ber = np.zeros((num_mac_iter))

rank_Q = np.zeros((num_tot_iter))


for gi in range(num_graph_iter):

    for qi in range(num_q_iter):

        species_traits = q_rs_all[qi]

        max_robots = 200    
        # generate a random end state
        random_transition = random_transition_matrix(num_nodes, max_rate/2)  # Divide max_rate by 2 for the random matrix to give some slack.                
        deploy_robots_init = np.random.randint(0, max_robots, size=(num_nodes, num_species))
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
        
        print "Calculating optimal transition matrix..."
        sys.stdout.flush()
        init_transition_values = np.array([])
        transition_m_init = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init, deploy_traits_desired,
                                                      species_traits, t_max, max_rate,l_norm, match, optimizing_t=True, force_steady_state=2.0)
        
        # -----------------------------------------------------------------------------#
        # Berman's method 
        print "Calculating optimal transition matrix, Berman..."
        sys.stdout.flush()
        transition_m_berman = Optimize_Berman(adjacency_m, deploy_robots_init, deploy_robots_final, species_traits,max_rate, max_time=t_max, verbose=True)
        
        
        # -----------------------------------------------------------------------------#
        # run microscopic stochastic simulation
        
        num_timesteps = int(t_max_sim / delta_t)
        
        deploy_robots_micro = np.zeros((num_nodes, num_timesteps, num_species, num_iter))
        deploy_robots_mic_ber = np.zeros((num_nodes, num_timesteps, num_species, num_iter))
        for it in range(num_iter):
            curr_mic_it = gi*num_q_iter+qi*num_iter+it
            
            print "Mic. iteration: ", curr_mic_it, " / ", num_mic_iter
            robots_new, deploy_robots_micro[:,:,:,it] = microscopic_sim(num_timesteps, delta_t, robots, deploy_robots_init, transition_m_init)
            robots_new_ber, deploy_robots_mic_ber[:,:,:,it] = microscopic_sim(num_timesteps, delta_t, robots, deploy_robots_init, transition_m_berman)
       
            t_min_mic[curr_mic_it] = get_traits_ratio_time(deploy_robots_micro[:,:,:,it], deploy_traits_desired, species_traits, match, min_ratio)
            t_min_mic_ber[curr_mic_it] = get_traits_ratio_time(deploy_robots_mic_ber[:,:,:,it], deploy_traits_desired, species_traits, match, min_ratio)
    
       
        # -----------------------------------------------------------------------------#
        # euler integration
        curr_mac_it = gi*num_q_iter+qi        
        
        rank_Q[curr_mac_it] = np.linalg.matrix_rank(species_traits)

        deploy_robots_mac_ber = run_euler_integration(deploy_robots_init, transition_m_berman, t_max_sim, delta_t)
        deploy_robots_euler = run_euler_integration(deploy_robots_init, transition_m_init, t_max_sim, delta_t)
        
        t_min_mac_ber[curr_mac_it] = get_traits_ratio_time(deploy_robots_mac_ber, deploy_traits_desired, species_traits, match, min_ratio)
        t_min_mac[curr_mac_it] = get_traits_ratio_time(deploy_robots_euler, deploy_traits_desired, species_traits, match, min_ratio)

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
    pickle.dump(deploy_robots_micro, open(prefix+"deploy_robots_micro.p", "wb"))
    pickle.dump(deploy_robots_euler, open(prefix+"deploy_robots_euler.p", "wb"))
    pickle.dump(t_min_mic, open(prefix+"t_min_mic.p", "wb"))
    pickle.dump(t_min_mac, open(prefix+"t_min_mac.p", "wb"))
    pickle.dump(t_min_mic_ber, open(prefix+"t_min_mic_ber.p", "wb"))
    pickle.dump(t_min_mac_ber, open(prefix+"t_min_mac_ber.p", "wb"))
    pickle.dump(rank_Q, open(prefix+"rank_Q.p", "wb"))

# -----------------------------------------------------------------------------#
# plots

# plot traits ratio
fig1 = plot_traits_ratio_time_micmac(deploy_robots_micro, deploy_robots_euler, deploy_traits_desired,species_traits, delta_t, match)

# plot time at which min ratio reached
fig2 = plot_t_converge_3(delta_t, t_min_mic, t_min_mac, t_min_mic_ber)


# -----------------------------------------------------------------------------#
# save plots
 
if save_plots:
    prefix = "./plots/" + run + "_"                        
    fig1.savefig(prefix+'micmicmac.eps') 
    fig2.savefig(prefix+'time_converge.eps') 


