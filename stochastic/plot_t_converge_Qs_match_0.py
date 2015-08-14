# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:34:37 2015
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
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
from simple_orrank import *
import funcdef_draw_network as nxmod


# match = 0 is 'at-least' cost function 
# runs: Q2, Q3, Q4
run = 'Q4'

# -----------------------------------------------------------------------------#
def plot_t_converge_OR_rk(delta_t, t_min_a, t_min_b, rks):

    num_graph_iter = t_min_a.shape[0]
    num_traits = t_min_a.shape[1]
    macro = (np.size(t_min_a.shape)==2)
    
    max_rk = int(max(rks))
    
    if not macro:
        el = t_min_a.shape[0] * t_min_a.shape[2]    
    else:
        el = t_min_a.shape[0]
        
    ta_f = []
    tb_f = [] 
    for i in range(max_rk):
        ta_f.append([])
        tb_f.append([])
        
    t_ab = []
    labels = []
    for gi in range(num_graph_iter):
        for ti in range(num_traits):
            rk = int(rks[gi*num_traits+ti])-1
            if not macro:
                ta_f[rk].append(delta_t * t_min_a[gi,ti,:].flatten())
                tb_f[rk].append(delta_t * t_min_b[gi,ti,:].flatten())
            else:
                ta_f[rk].append(delta_t * t_min_a[gi,ti].flatten())
                tb_f[rk].append(delta_t * t_min_b[gi,ti].flatten())
            
    for rk in range(max_rk):
        t_ab.append(ta_f[rk])
        t_ab.append(tb_f[rk])
        labels.append(str(rk+1))
    
    N = max_rk*2 + 2
    fig = plt.figure()
    ax = plt.gca()

    x = range(1,max_rk+1)
    x_pos = []
    for i in range(max_rk):
        x_pos.append(x[i]-0.15)
        x_pos.append(x[i]+0.15)
          
    bp = plt.boxplot(t_ab, positions=x_pos, widths=0.2, notch=0, sym='+', vert=1, whis=1.5) 
                     
                     
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.grid(axis='y')
    ymin = 0 
    ymax = 4
    ax.set_ylim([0, ymax])    
    ax.set_xlim([0.4, max_rk+0.6])


    plt.xticks(x, labels) #, rotation='vertical')

    return fig
    
    
# -----------------------------------------------------------------------------#
# load data

prefix = "./data/" + run + "/" + run + "_"


list_Q = pickle.load(open(prefix+"list_Q.p", "rb"))
t_min_mic = pickle.load(open(prefix+"t_min_mic.p", "rb"))
t_min_mac = pickle.load(open(prefix+"t_min_mac.p", "rb"))
t_min_mic_ber = pickle.load(open(prefix+"t_min_mic_ber.p", "rb"))
t_min_mac_ber = pickle.load(open(prefix+"t_min_mac_ber.p", "rb"))


# -----------------------------------------------------------------------------#
# get rank

ll = len(list_Q)
rks = np.zeros(ll)

for i in range(ll):
    q = list_Q[i]
    rks[i] = orrank(q)
    max_rk = int(max(rks))
    

# -----------------------------------------------------------------------------#
# plot
delta_t = 0.04


fig1 = plot_t_converge_OR_rk(delta_t, t_min_mic, t_min_mic_ber, rks)
plt.axes().set_aspect(0.5,'box')
plt.show()

fig2 = plot_t_converge_OR_rk(delta_t, t_min_mac, t_min_mac_ber, rks)
plt.axes().set_aspect(0.5,'box')
plt.show()





