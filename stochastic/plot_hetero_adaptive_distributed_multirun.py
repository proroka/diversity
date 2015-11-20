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
#runs = ['D32','D33','D34', 'D35'] 
#runs = ['D48','D49','D50', 'D51']
#runs = ['D44','D45','D46', 'D47'] # good
#runs = ['D36','D37','D38', 'D39'] # good
#runs = ['D40','D41','D42', 'D43'] 

#runs = ['D52','D53','D54', 'D55'] 
#runs = ['D56','D57', 'D59'] # D58 not done yet
#runs = ['D60','D62', 'D63'] # D61 not done yet 
#runs = ['D64','D65','D66'] #D67 not done yet 

#runs = ['D68','D69','D70','D71'] 

# 50/50 init/final, numts=20, with SS
runs = ['D52','D53','D54', 'D55','D68','D69','D70','D71'] 

save_plots = False
plot_graph = False

delta_t = 0.04 # time step
match = 1
min_ratio = 0.1


diff_ratio_mic_list = []

for i in range(len(runs)):
    prefix = "./data/" + runs[i] + "/" + runs[i] + '_'
    
    graph = pickle.load(open(prefix+"graph.p", "rb"))
    species_traits = pickle.load(open(prefix+"species_traits.p", "rb"))
    deploy_robots_init = pickle.load(open(prefix+"deploy_robots_init.p", "rb"))
    deploy_traits_init = pickle.load(open(prefix+"deploy_traits_init.p", "rb"))
    deploy_traits_desired = pickle.load(open(prefix+"deploy_traits_desired.p", "rb"))
    deploy_robots_micro_adapt = pickle.load(open(prefix+"deploy_robots_micro_adapt.p", "rb"))
    deploy_robots_micro = pickle.load(open(prefix+"deploy_robots_micro.p", "rb"))
    deploy_robots_euler = pickle.load(open(prefix+"deploy_robots_euler.p", "rb"))
    deploy_robots_micro_adapt_hop = pickle.load(open(prefix+"deploy_robots_micro_adapt_hop.p", "rb"))

    # returns: diffmic_rat = np.zeros((num_tsteps, num_it, num_hops)) 
    diff_ratio_mic = get_traits_ratio_time_mic_distributed(deploy_robots_micro_adapt_hop, deploy_traits_desired, species_traits, match)

    diff_ratio_mic_list.append(diff_ratio_mic)

# concatenate arrays
diff_ratio_mic_stack = np.concatenate(diff_ratio_mic_list, axis=1)

fig = plot_traits_ratio_time_mic_distributed_multirun(diff_ratio_mic_stack, delta_t)

  

#------------------------------------------
# Boxplots

df = diff_ratio_final = diff_ratio_mic_stack[-20,:,:]

fig = plt.figure()
ax = plt.gca()
x = range(df.shape[1])
bp = plt.boxplot(df, positions=x, widths=0.2, notch=0, sym='+', vert=1, whis=1.5) 

ymin = 0 
ymax = 0.15
ax.set_ylim([0, ymax])    
#ax.set_xlim([1, num_hops])

plt.axes().set_aspect(4.5/0.14,'box')   
plt.xlabel('Hops')    
plt.ylabel('Ratio of misplaced traits')
    
plt.show()




