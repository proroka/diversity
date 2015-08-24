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
from generate_Q import *
from simple_orrank import *

# -----------------------------------------------------------------------------#

# 1. create graph
# 2. initialize robot deployment on nodes
# 3. initialize transition matrix
# 4. run macro-discrete model  / micro simulator
# 5. plot evolution of robot population per node

# -----------------------------------------------------------------------------#
# initialize world and robot community

save_data = True
save_plots = True

tstart = time.strftime("%Y%m%d-%H%M%S")

# simulation parameters
t_max = 10.0 # influences desired state and optmization of transition matrix
t_max_sim = 10.0 # influences simulations and plotting
delta_t = 0.04 # time step
max_rate = 2.0 # Maximum rate possible for K.


# -----------------------------------------------------------------------------#
# initialize system

run = 'Q34'

# goal function
match = 0 # 1: exact 0: at-least
if match==1:
    berman = True
else:
    berman = False

num_nodes = 8
num_iter = 1 # micro sim
num_graph_iter = 60 # random graphs

num_species = 6
num_q_iter = num_species # num_traits from 1 to num_species
#range_q_iter = np.array([0, 1, 2, 3]) # num_traits-1
range_q_iter = range(num_q_iter)

# cost function
l_norm = 2 # 2: quadratic 1: absolute

# eval convergence time 
use_strict = True
strict_slack = 1.4 # max 1.4*err on desired robot distrib must be true for trait distrib to be valid 
match_margin = 0.02 # used when match=0 


# -----------------------------------------------------------------------------#
# find time at which min ratio is found

num_mic_iter = num_iter * num_graph_iter * num_q_iter
num_mac_iter = num_graph_iter * num_q_iter

min_ratio = 0.05
t_min_mic = np.zeros((num_graph_iter, num_q_iter, num_iter))
t_min_mic_ber = np.zeros((num_graph_iter, num_q_iter, num_iter))
t_min_mac = np.zeros((num_graph_iter, num_q_iter))
t_min_mac_ber = np.zeros((num_graph_iter, num_q_iter))

rank_Q = np.zeros((num_graph_iter, num_q_iter))
list_Q = []

for gi in range(num_graph_iter):

    #for qi in range(num_q_iter):
    for qi in range_q_iter:

        rk = 0
        num_traits = qi+1

        print "Graph: ", gi, "rank: ", num_traits
        if match==1:
            while rk != num_traits:
                species_traits, rk, s = generate_Q(num_species, num_traits)
        else:
            species_traits = generate_matrix_with_orrank(num_species, num_species, num_traits)
        list_Q.append(species_traits)

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
            deploy_traits_desired *= (np.random.rand()*match_margin + (1.0-match_margin))
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
        if use_strict:
            # Get final robot distribution.
            strict_deploy_robots_final = run_euler_integration(deploy_robots_init, transition_m_init, t_max_sim, delta_t)[:,-1,:]

        # -----------------------------------------------------------------------------#
        # Berman's method
        if berman:
            print "Calculating optimal transition matrix, Berman..."
            sys.stdout.flush()
            transition_m_berman = Optimize_Berman(adjacency_m, deploy_robots_init, deploy_robots_final, species_traits,max_rate, max_time=t_max, verbose=True)


        # -----------------------------------------------------------------------------#
        # run microscopic stochastic simulation

        num_timesteps = int(t_max_sim / delta_t)

        for it in range(num_iter):
            t_min_mic[gi,qi,it] = microscopic_sim_get_time(num_timesteps, delta_t, robots, deploy_robots_init, transition_m_init, deploy_traits_desired, strict_deploy_robots_final if use_strict else None, species_traits, match, min_ratio, strict_slack)
            if berman:
                t_min_mic_ber[gi,qi,it] = microscopic_sim_get_time(num_timesteps, delta_t, robots, deploy_robots_init, transition_m_berman, deploy_traits_desired, deploy_robots_final if use_strict else None, species_traits, match, min_ratio, strict_slack)

        # -----------------------------------------------------------------------------#
        # euler integration

        rank_Q[gi,qi] = np.linalg.matrix_rank(species_traits)

        if berman:
            deploy_robots_mac_ber = run_euler_integration(deploy_robots_init, transition_m_berman, t_max_sim, delta_t)
        deploy_robots_euler = run_euler_integration(deploy_robots_init, transition_m_init, t_max_sim, delta_t)

        if berman:
            if not use_strict:
                t_min_mac_ber[gi,qi] = get_traits_ratio_time(deploy_robots_mac_ber, deploy_traits_desired, species_traits, match, min_ratio)
            else:
                t_min_mac_ber[gi,qi] = get_traits_ratio_time_strict(deploy_robots_mac_ber, deploy_traits_desired, deploy_robots_final, species_traits, match, min_ratio, strict_slack)
        t_min_mac[gi,qi] = get_traits_ratio_time(deploy_robots_euler, deploy_traits_desired, species_traits, match, min_ratio)

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
    #pickle.dump(deploy_robots_micro, open(prefix+"deploy_robots_micro.p", "wb"))
    pickle.dump(deploy_robots_euler, open(prefix+"deploy_robots_euler.p", "wb"))
    pickle.dump(t_min_mic, open(prefix+"t_min_mic.p", "wb"))
    pickle.dump(t_min_mac, open(prefix+"t_min_mac.p", "wb"))

    pickle.dump(rank_Q, open(prefix+"rank_Q.p", "wb"))
    pickle.dump(list_Q, open(prefix+"list_Q.p", "wb"))

    if berman:
        pickle.dump(t_min_mic_ber, open(prefix+"t_min_mic_ber.p", "wb"))
        pickle.dump(t_min_mac_ber, open(prefix+"t_min_mac_ber.p", "wb"))

# -----------------------------------------------------------------------------#
# plots

if berman:
    # flatten for plotting function
    for rk in range(num_species):
        prefix = "./plots/" + run + "_rank" + str(rk+1)+"_"
        t_mic_f = t_min_mic[:,rk,:].flatten()
        t_ber_f = t_min_mic_ber[:,rk,:].flatten()
        t_mac_f = t_min_mac[:,rk].flatten()

        fig = plot_t_converge_3(delta_t, t_mic_f, t_mac_f, t_ber_f)
        fig.savefig(prefix+'time_converge.eps')

