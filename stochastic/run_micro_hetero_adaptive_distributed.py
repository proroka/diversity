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
sys.path.append('plotting')
sys.path.append('utilities')
from optimize_transition_matrix_hetero import *
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
# utility functions for x-hop neighborhood

# returns a list of neighbors for each node 
def GetNeighbors(A, n_hops):
    t = np.identity(A.shape[0])
    for _ in range(n_hops):
        t += t.dot(A)
    neighbors = []
    for i in range(A.shape[0]):
        neighbors.append(np.where(t[i,:] > 0)[0].tolist())
    return neighbors


def BuildLocalDistribution(B, node_index, neighbors):
    num_nodes = B.shape[0]
    deploy = B.copy()
    # Set to zero all neighbors.
    deploy[neighbors[node_index],:] = 0
    # Average all robots over the non-neighboring nodes.
    uniform = np.ones((num_nodes, num_species)) * (np.sum(deploy, 0) / float(num_nodes - len(neighbors[node_index])))
    # Put the original robot counts in the neighboring nodes.
    uniform[neighbors[node_index],:] = B[neighbors[node_index],:]
    return uniform



# -----------------------------------------------------------------------------#
# initialize world and robot community

run = 'D79'

load_data = False
load_run = 'M18'
load_prefix = "./data/" + load_run + '/' + load_run + "_"

save_data = True

# random initial and final trait distributions
fix_init = True
fix_final = True

distributed = True

tstart = time.strftime("%Y%m%d-%H%M%S")
print str(run)
print "Time start: ", tstart

# simulation parameters
t_max = 8.0 # influences desired state and optmization of transition matrix
t_max_sim = 7.0 # influences simulations and plotting
num_iter = 5 # iterations of micro sim
delta_t = 0.04 # time step
max_rate = 1.0 # Maximum rate possible for K.

# adaptive
numts_window = 175  # number of time steps per window for adaptive opt.
FSS = 2.0 #numts_window * delta_t # force steady state, et to 0.0 if not used

# graph
num_nodes = 8
half_num_nodes = num_nodes / 2    

# cost function
l_norm = 2 # 2: quadratic 1: absolute
match = 1 # 1: exact 0: at-least
match_margin = 0.2 # used when match=0 

# robot species
total_num_robots = 1000.0 
num_species = 4
num_traits = 5

# distributed belief distribution
num_hops = 5
if not distributed:
    num_hops = 1


# -----------------------------------------------------------------------------#
# initialize distributions

    
# generate random robot species with fixed diversity
list_Q = []

if not load_data:
    rk = 0
    while rk != num_species:
        species_traits, rk, s = generate_Q(num_species, num_traits)
else:
    species_traits = pickle.load(open(load_prefix+"species_traits.p", "rb"))
    print species_traits    
    

list_Q.append(species_traits)
    
# generate a random end state
random_transition = random_transition_matrix(num_nodes, max_rate/2)  # Divide max_rate by 2 for the random matrix to give some slack.
deploy_robots_init = np.random.randint(0, 100, size=(num_nodes, num_species))

if fix_init:
    deploy_robots_init[half_num_nodes:,:] = 0
    # normalize
    deploy_robots_init = deploy_robots_init * total_num_robots / np.sum(np.sum(deploy_robots_init, axis=0))
    sum_species = np.sum(deploy_robots_init,axis=0)
if load_data:
    deploy_robots_init = pickle.load(open(load_prefix+"deploy_robots_init.p", "rb"))
    sum_species = np.sum(deploy_robots_init,axis=0)
        
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
# initialize micro

num_timesteps = int(t_max_sim / delta_t)
t_window = float(numts_window) * delta_t
slices = int(t_max_sim / t_window)

deploy_robots_micro = np.zeros((num_nodes, num_timesteps, num_species, num_iter))

init_transition_values = np.array([])
# distributed version of transition matrix
transition_m_d = np.zeros((num_nodes, num_nodes, num_species, num_nodes))
# distributed version of deployment beliefs
deploy_robots_init_slice_d_all = np.zeros((num_nodes, num_species, num_nodes))

deploy_robots_micro_adapt_hop = np.zeros((num_nodes, num_timesteps, num_species, num_iter, num_hops))

# -----------------------------------------------------------------------------#
# initialize graph

if not load_data:
    graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.6)
else:
    graph = pickle.load(open(load_prefix+"graph.p", "rb"))

# get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))

# find optimal transition matrix for plain micro
init_transition_values = np.array([])

#print init_transition_values
#print adjacency_m
#print deploy_robots_init
#print deploy_traits_desired
#print species_traits
#sys.stdout.flush()

transition_m_init = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init, deploy_traits_desired,
                                              species_traits, t_max, max_rate,l_norm, match, optimizing_t=True, force_steady_state=4.0)



# -----------------------------------------------------------------------------#
# run microscopic stochastic simulation

if not distributed:

    deploy_robots_micro = np.zeros((num_nodes, num_timesteps, num_species, num_iter))
    for it in range(num_iter):
        print "Iteration: ", it
        robots_new, deploy_robots_micro[:,:,:,it] = microscopic_sim(num_timesteps, delta_t, robots, deploy_robots_init, transition_m_init)


# -----------------------------------------------------------------------------#
# run simulation

for it in range(num_iter):

    for nh in range(num_hops):
        neighbor_nodes = GetNeighbors(adjacency_m, nh)
        #for it in range(num_iter):

        # initialize structure for transition matrices
        for nd in range(num_nodes):
            transition_m_d[:,:,:,nd] = transition_m_init.copy()
        init_transition_values = transition_m_init.copy() ### TODO
    
        deploy_robots_init_slice = deploy_robots_init.copy()
        robots_slice = robots.copy()
        # initialize micro structure
        deploy_robots_micro_adapt = np.zeros((num_nodes, num_timesteps, num_species))
        
        for sl in range(slices):
            print "Hop: ", nh, "Iteration: ", it+1 , "/", num_iter, "Slice: ", sl+1,"/", slices
            
            if(distributed):
                # adaptive, distributed:
                # for robots at same task, compute belief of robot distribution and optimize transition matrix
                for nd in range(num_nodes):   
                    # create naive belief of robot distribution: uniform distrib outside this node             
                    deploy_robots_init_slice_d = BuildLocalDistribution(deploy_robots_init_slice, nd, neighbor_nodes)
                    
                    # optimize transition matrix for this distribution belief
                    transition_m_d[:,:,:,nd] = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_slice_d, deploy_traits_desired,
                                                                        species_traits, t_max, max_rate,l_norm, match, optimizing_t=True,
                                                                        force_steady_state=FSS)            
                    
                # run micro-dstributed with multiple K (1 for each node): robots at same node only use same transition matrix
                robots_slice, deploy_robots_micro_slice = microscopic_sim_distrib(numts_window + 1, delta_t, robots_slice, deploy_robots_init_slice, transition_m_d)
                
            else:
                transition_m = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_slice, deploy_traits_desired,
                                                 species_traits, t_max, max_rate,l_norm, match, optimizing_t=True,
                                                 force_steady_state=FSS)
                robots_slice, deploy_robots_micro_slice = microscopic_sim(numts_window + 1, delta_t, robots_slice, deploy_robots_init_slice, transition_m)

            deploy_robots_init_slice = deploy_robots_micro_slice[:,-1,:] # final time-step of micro is init distrib for next optim.
            # put together slices
            deploy_robots_micro_adapt[:,sl*numts_window:(sl+1)*numts_window,:] = deploy_robots_micro_slice[:,:-1,:]
                 
        # Simulate the last extra bit.
        timesteps_left = num_timesteps - slices * numts_window
        assert timesteps_left >= 0
        if timesteps_left > 0:
            robots_slice, deploy_robots_micro_slice = microscopic_sim_distrib(timesteps_left + 1, delta_t, robots_slice, deploy_robots_init_slice, transition_m_d)
            deploy_robots_micro_adapt[:,-timesteps_left:,:] = deploy_robots_micro_slice[:,:-1,:]
                              
        deploy_robots_micro_adapt_hop[:,:,:,it,nh] =  deploy_robots_micro_adapt.copy()                                          
       

# -----------------------------------------------------------------------------#
# euler integration

deploy_robots_euler = run_euler_integration(deploy_robots_init, transition_m_init, t_max_sim, delta_t)

# -----------------------------------------------------------------------------#
# save variables

if save_data:
    
    tend = time.strftime("%Y%m%d-%H%M%S")
   
    prefix = "./data/" + run + "_"
    print str(run)    
    print "Time start: ", tstart
    print "Time end: ", tend
    
    pickle.dump(species_traits, open(prefix+"species_traits.p", "wb"))
    pickle.dump(graph, open(prefix+"graph.p", "wb"))
    pickle.dump(deploy_robots_init, open(prefix+"deploy_robots_init.p", "wb"))
    pickle.dump(deploy_traits_init, open(prefix+"deploy_traits_init.p", "wb"))
    pickle.dump(deploy_traits_desired, open(prefix+"deploy_traits_desired.p", "wb"))
    pickle.dump(deploy_robots_micro_adapt, open(prefix+"deploy_robots_micro_adapt.p", "wb"))
    pickle.dump(deploy_robots_micro, open(prefix+"deploy_robots_micro.p", "wb"))
    pickle.dump(deploy_robots_euler, open(prefix+"deploy_robots_euler.p", "wb"))
    pickle.dump(deploy_robots_micro_adapt_hop, open(prefix+"deploy_robots_micro_adapt_hop.p", "wb"))




