# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:59:45 2015
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
import funcdef_draw_network as nxmod


run = 'Q3'
save_plots = False

# -----------------------------------------------------------------------------#
def plot_t_converge(delta_t, t_min_a, t_min_b):

    num_rk = t_min_a.shape[1]
    macro = (np.size(t_min_a.shape)==2)
    
    if not macro:
        el = t_min_a.shape[0] * t_min_a.shape[2]    
    else:
        el = t_min_a.shape[0]
        
    ta_f = np.zeros((num_rk, el))
    tb_f = np.zeros((num_rk, el))
    t_ab = []
    labels = []
    for rk in range(num_rk):
        if not macro:
            ta_f[rk,:] = delta_t * t_min_a[:,rk,:].flatten()
            tb_f[rk,:] = delta_t * t_min_b[:,rk,:].flatten()
        else:
            ta_f[rk,:] = delta_t * t_min_a[:,rk].flatten()
            tb_f[rk,:] = delta_t * t_min_b[:,rk].flatten()
        t_ab.append(ta_f[rk,:])
        t_ab.append(tb_f[rk,:])
        labels.append(str(rk+1))
    
    N = num_rk*2 + 2
    fig = plt.figure()
    ax = plt.gca()

    x = range(1,num_rk+1)
    print x
    x_pos = []
    for i in range(num_rk):
        x_pos.append(x[i]-0.15)
        x_pos.append(x[i]+0.15)
        
    print x_pos    
    bp = plt.boxplot(t_ab, positions=x_pos, widths=0.2, notch=0, sym='+', vert=1, whis=1.5) 
                     
                     
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.grid(axis='y')
    ymin = 0 
    ymax = 8 
    ax.set_ylim([0, ymax])    
    ax.set_xlim([0.4, num_rk+0.6])

    
    #plt.legend(loc='upper right', shadow=False, fontsize='large')     
    #plt.ylabel('T')    
    

    plt.xticks(x, labels) #, rotation='vertical')

    return fig
    

    
# -----------------------------------------------------------------------------#
# load data

prefix = "./data/" + run + "/" + run + "_"
#prefix = "./data/" + run + "_"

#rank_Q = pickle.load(open(prefix+"rank_Q.p", "rb"))
t_min_mic = pickle.load(open(prefix+"t_min_mic.p", "rb"))
t_min_mac = pickle.load(open(prefix+"t_min_mac.p", "rb"))
t_min_mic_ber = pickle.load(open(prefix+"t_min_mic_ber.p", "rb"))
t_min_mac_ber = pickle.load(open(prefix+"t_min_mac_ber.p", "rb"))

# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#
# plot
delta_t = 0.04


fig1 = plot_t_converge(delta_t, t_min_mic, t_min_mic_ber)
plt.axes().set_aspect(0.5,'box')
plt.show()

fig1 = plot_t_converge(delta_t, t_min_mac, t_min_mac_ber)
plt.axes().set_aspect(0.5,'box')
plt.show()


# -----------------------------------------------------------------------------#
# save plots
 
#if save_plots:
    #fig1.savefig(prefix+'rank3_t_conv.eps') 
                        



