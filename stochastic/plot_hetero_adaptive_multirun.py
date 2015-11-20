# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:40:29 2015
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
sys.path.append('plotting')
sys.path.append('utilities')
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod
from generate_Q import *
from simple_orrank import *


#load data
runs = ['M01','M02','M03', 'M05'] 
# G2
runs = ['M11','M12','M13', 'M14', 'M15'] 
# G1
runs = ['M16','M17','M18', 'M19', 'M20'] 

save_plots = False
plot_graph = False

delta_t = 0.04 # time step
match = 1
min_ratio = 0.1

diff_ratio_mic_adp_list = []
diff_ratio_mic_list = []

for i in range(len(runs)):
    prefix = "./data/" + runs[i] + "/" + runs[i] + '_'
        
    # Load data
    graph = pickle.load(open(prefix+"graph.p", "rb"))
    species_traits = pickle.load(open(prefix+"species_traits.p", "rb"))
    deploy_robots_init = pickle.load(open(prefix+"deploy_robots_init.p", "rb"))
    deploy_traits_init = pickle.load(open(prefix+"deploy_traits_init.p", "rb"))
    deploy_traits_desired = pickle.load(open(prefix+"deploy_traits_desired.p", "rb"))
    deploy_robots_micro_adapt = pickle.load(open(prefix+"deploy_robots_micro_adapt.p", "rb"))
    deploy_robots_micro = pickle.load(open(prefix+"deploy_robots_micro.p", "rb"))
    deploy_robots_euler = pickle.load(open(prefix+"deploy_robots_euler.p", "rb"))
    deploy_robots_micro_adapt_hop = pickle.load(open(prefix+"deploy_robots_micro_adapt_hop.p", "rb"))

    deploy_robots_micro_adapt_h1 = deploy_robots_micro_adapt_hop[:,:,:,:,0]

    # returns: diffmic_rat = np.zeros((num_tsteps, num_it, num_hops)) 

    # micro adaptive (1-hop only, full graph knowledge)
    diff_ratio_mic_adp_t = get_traits_ratio_time_mic_distributed(deploy_robots_micro_adapt_hop, deploy_traits_desired, species_traits, match)
    diff_ratio_mic_adp = diff_ratio_mic_adp_t[:,:,0]    
    # plain micro
    diff_ratio_mic = get_traits_ratio_time_mic(deploy_robots_micro_adapt_h1, deploy_traits_desired, species_traits, match)


    diff_ratio_mic_adp_list.append(diff_ratio_mic_adp)
    diff_ratio_mic_list.append(diff_ratio_mic)

# concatenate arrays
diff_ratio_mic_stack = np.concatenate(diff_ratio_mic_list, axis=1)
diff_ratio_mic_adp_stack = np.concatenate(diff_ratio_mic_adp_list, axis=1)


#fig = plot_traits_ratio_time_micmicmac(deploy_robots_micro, deploy_robots_micro_adapt_h1, deploy_robots_euler, 
#                                     deploy_traits_desired, species_traits, delta_t, match)


fig = plot_traits_ratio_time_micmic_multirun(diff_ratio_mic_stack, diff_ratio_mic_adp_stack, delta_t)



#---------------------------------------------------
# plot graph
if plot_graph:
    plt.axis('equal')
    fig1 = nxmod.draw_circular(deploy_traits_init, graph,linewidths=3)
    plt.show()