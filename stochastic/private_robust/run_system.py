# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:47:07 2016
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
sys.path.append('../plotting')
sys.path.append('../utilities')
sys.path.append('..')
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod
from generate_Q import *
from funcdef_util_privacy import *

# -----------------------------------------------------------------------------#

# 1. setup: create graph
# 2. setup: initialize robot deployment on nodes
# 3. setup: initialize transition matrix
# 4. optimize transition rates
#
# -----------------------------------------------------------------------------#
# initialize world and robot community

run = 'RC06'

selected_runs = True # run for selected parameter range

load_data = True
load_run = 'RC00'
load_prefix = "../data/RCx/" + load_run + '/' + load_run + "_"
save_data = True

# random initial and final trait distributions
fix_init = False
fix_final = False

plot_graph = False
plot_run = True

tstart = time.strftime("%Y%m%d-%H%M%S")
tic = time.time()
print str(run)
print "Time start: ", tic #tstart

# simulation parameters
t_max = 12.0 # influences desired state and optmization of transition matrix
t_max_sim = 12.0 # influences simulations and plotting
delta_t = 0.02 # time step
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

range_alpha = np.linspace(0, 1, 5); range_alpha[0] = 0.01
range_beta = np.linspace(5, 0, 5); range_beta[-1] = 0.01
range_lambda = np.array([0.001, 0.5, 1.0, 2.0, 4.0])
optimize_t = True
num_sample_iter = 5

testing = True
if testing:
   range_alpha = np.array([1.0]) #np.linspace(0, 1, 5); range_alpha[0] = 0.01
   range_beta = np.array([0.01]) #np.linspace(5, 0, 5); range_beta[-1] = 0.01
   range_lambda = np.array([10.0]) #np.array([0.001, 0.5, 1.0, 2.0, 4.0])
   optimize_t = True
   num_sample_iter = 3 



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

random_transition = random_transition_matrix(num_nodes, max_rate/2)  # Divide max_rate by 2 for the random matrix to give some slack.


print 'Getting high enough starting error...'
sys.stdout.flush()

ratio_init = 0.0
while ratio_init < 0.3 and not load_data:
    # sample initial
    print ratio_init
    deploy_robots_init = np.random.randint(0, 100, size=(num_nodes, num_species))
    # normalize
    deploy_robots_init = deploy_robots_init * total_num_robots / np.sum(np.sum(deploy_robots_init, axis=0))
    sum_species = np.sum(deploy_robots_init,axis=0)
    deploy_traits_init = np.dot(deploy_robots_init, species_traits)    
    # sample final desired trait distribution based on random transition matrix
    deploy_robots_final = sample_final_robot_distribution(deploy_robots_init, random_transition, t_max*4., delta_t)
    deploy_traits_desired = np.dot(deploy_robots_final, species_traits)
    ratio_init_mat = np.abs(np.dot(deploy_robots_init, species_traits) - deploy_traits_desired)
    ratio_init = np.sum(ratio_init_mat) / np.sum(np.sum(deploy_traits_init)) / 2.0

if load_data:
    deploy_robots_init = pickle.load(open(load_prefix+"deploy_robots_init.p", "rb"))
    sum_species = np.sum(deploy_robots_init,axis=0)

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


num_timesteps = int(t_max_sim / delta_t)

traj_ratio = {}

opt_time = []

for el in range(len(range_lambda)):
    for a in range(len(range_alpha)):
        b = a        
        lap = range_lambda[el]
        alpha = range_alpha[a]
        beta = range_beta[b] 
    
        temp_r = el*len(range_alpha) + a
        print '****** Run ', temp_r, ' / ', len(range_lambda)*len(range_alpha)
    
        ratio = np.zeros((num_timesteps, num_sample_iter))
        deploy_robots_euler_it = np.zeros((num_nodes, num_timesteps, num_species, num_sample_iter))
        for i in range(num_sample_iter):
        # optimize based on noisy initial state
    
            
            
            if lap > 0.001:
                lap_val = np.random.laplace(loc=0.0, scale=lap, size=(num_nodes, num_species))
                deploy_robots_init_noisy = deploy_robots_init + lap_val
                temp0 = np.sum(deploy_robots_init,axis=0) 
                temp1 = np.sum(deploy_robots_init_noisy,axis=0) 
                #print 'Orig. Species Sum = ', temp0, 'Total', sum(temp0)             
                #print 'Noisy Species Sum = ', temp1, 'Total', sum(temp1)  
                # normalize robot distribution so that correct robots per species available
                ts = np.sum(deploy_robots_init_noisy, axis=0)
                for s in range(num_species):
                    deploy_robots_init_noisy[:,s] = deploy_robots_init_noisy[:,s] / float(ts[s]) * float(sum_species[s])
                    
                #temp2 = np.sum(deploy_robots_init_noisy,axis=0) 
                #print 'Norm. Species Sum = ', temp2, 'Total', sum(temp2)  
                
            else:
                deploy_robots_init_noisy = deploy_robots_init.copy()
    
            temp3 = np.sum(deploy_robots_init_noisy,axis=0) 
            print 'Norm. Species Sum = ', temp3, 'Total', sum(temp3) 
            
            init_transition_values = np.array([])
    
            otic = time.time()
            print 'Optimizing rates...'
            sys.stdout.flush()
            transition_m_init = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_noisy, deploy_traits_desired,
                                                      species_traits, t_max, max_rate,l_norm, match, optimizing_t=optimize_t, force_steady_state=4.0, alpha=alpha, beta=beta)
                                                      
            otoc = time.time()
            print '*** Time for opt.: ', otoc - otic
            opt_time.append(otoc - otic)            
                             
            # run euler integration, evaluate trajectory based on true initial state with K from noisy optimization above
            deploy_robots_euler_it[:,:,:,i] = run_euler_integration(deploy_robots_init, transition_m_init, t_max_sim, delta_t)
                        
            
            # store error ratio
            ratio[:,i] = get_traits_ratio_time_traj(deploy_robots_euler_it[:,:,:,i], deploy_traits_desired, species_traits, match)
            
            
        traj_ratio[(el, a, b)] = ratio #.copy()

# -----------------------------------------------------------------------------#
# plot simulations

def GetColors(n):
    cm = plt.get_cmap('gist_rainbow')
    return [cm(float(i) / float(n)) for i in range(n)]

if plot_run:
    colors = GetColors(len(range_alpha))

    for el in range(len(range_lambda)):
        fig = plt.figure(figsize=(5,4))        
        lines = []
        legends = []
        for a in range(len(range_alpha)):
            b = a
            col = colors[a]
            lap = range_lambda[el]
            alpha = range_alpha[a]
            beta = range_beta[b]
            tr = traj_ratio[(el, a, b)]
            t = np.arange(tr.shape[0]) * delta_t
            m = np.mean(tr, axis=1)
            s = np.std(tr, axis=1)
            legends.append(('alpha = %.2f, beta = %.2f' % (alpha, beta)))
            lines.append(plt.plot(t, m, c=col, lw=2)[0])
            plt.plot(t, tr, c=col)
            plt.fill_between(t, m - s, m + s, facecolor=col, alpha=0.3)
        plt.title('lap = %.2f' % lap)
        plt.ylim([0, 0.5])
        plt.legend(lines, legends)
        plt.show()
        fig.savefig('lap = %.2f.eps' % lap)
        


# -----------------------------------------------------------------------------#
# save variables

if save_data:

    tend = time.strftime("%Y%m%d-%H%M%S")
    toc = time.time()
    total_time = toc - tic
    
    prefix = "../data/RCx/" + run + "_"
    print str(run)
    print "Time start: ", tic #tstart
    print "Time end: ", toc # tend
    print "Total sim. time: ", total_time

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
    pickle.dump(opt_time, open(prefix+"opt_time.p", "wb"))
    pickle.dump(total_time, open(prefix+"total_time.p", "wb"))


