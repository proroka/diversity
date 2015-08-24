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


# use_strict=0: Q5 
# use_strict=1, only Berman, factor 1.0: Q10
# use_strict=1, on both mic sims, f=1.3: Q22
# ORrank: Q9
run = 'Q32'

match = False
if match:
    berman = True
else:
    berman = False

save_plots = True

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
    x_pos = []
    for i in range(num_rk):
        x_pos.append(x[i]-0.15)
        x_pos.append(x[i]+0.15)
          
    bp = plt.boxplot(t_ab, positions=x_pos, widths=0.2, notch=0, sym='+', vert=1, whis=1.5) 
                     
                     
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.grid(axis='y')
    ymin = 0 
    ymax = 12
    ax.set_ylim([0, ymax])    
    ax.set_xlim([0.4, num_rk+0.6])


    plt.xticks(x, labels) #, rotation='vertical')

    return fig
    
# -----------------------------------------------------------------------------#
def plot_t_converge_shaded(delta_t, t_min_a, t_min_b=None):

    num_rk = t_min_a.shape[1]
    macro = (np.size(t_min_a.shape)==2)
    
    if not macro:
        el = t_min_a.shape[0] * t_min_a.shape[2]    
    else:
        el = t_min_a.shape[0]
        
    ta_f = np.zeros((num_rk, el))
    tb_f = np.zeros((num_rk, el))
    pc_a = np.zeros((num_rk,3))
    pc_b = np.zeros((num_rk,3))
    p0 = 25
    p1 = 50
    p2 = 75
    labels = []
    
    if t_min_b is None:
        for rk in range(num_rk):
            if not macro:
                ta_f[rk,:] = delta_t * t_min_a[:,rk,:].flatten()
            else:
                ta_f[rk,:] = delta_t * t_min_a[:,rk].flatten()
            pc_a[rk,0] = np.percentile(ta_f[rk,:],p0)
            pc_a[rk,1] = np.percentile(ta_f[rk,:],p1)
            pc_a[rk,2] = np.percentile(ta_f[rk,:],p2)
            labels.append(str(rk+1))
    else:
        for rk in range(num_rk):
            if not macro:
                ta_f[rk,:] = delta_t * t_min_a[:,rk,:].flatten()
                tb_f[rk,:] = delta_t * t_min_b[:,rk,:].flatten()
            else:
                ta_f[rk,:] = delta_t * t_min_a[:,rk].flatten()
                tb_f[rk,:] = delta_t * t_min_b[:,rk].flatten()
            pc_a[rk,0] = np.percentile(ta_f[rk,:],p0)
            pc_a[rk,1] = np.percentile(ta_f[rk,:],p1)
            pc_a[rk,2] = np.percentile(ta_f[rk,:],p2)
            pc_b[rk,0] = np.percentile(tb_f[rk,:],p0)
            pc_b[rk,1] = np.percentile(tb_f[rk,:],p1)
            pc_b[rk,2] = np.percentile(tb_f[rk,:],p2)
            labels.append(str(rk+1))
    
    N = num_rk*2 + 2
    fig = plt.figure()
    ax = plt.gca()

    x = range(1,num_rk+1)
    x_pos = []
    for i in range(num_rk):
        x_pos.append(x[i]-0.15)
        x_pos.append(x[i]+0.15)
          
    
    
    l1 = plt.plot(x, pc_a[:,1], color='green', linewidth=2, label='explicit')
    plt.fill_between(x, pc_a[:,2], pc_a[:,0], facecolor='green', alpha=0.3)
    if t_min_b is not None:    
        l2 = plt.plot(x, pc_b[:,1], color='red', linewidth=2, label='implicit')
        plt.fill_between(x, pc_b[:,2], pc_b[:,0], facecolor='red', alpha=0.3)
        
    plt.legend(loc='upper left', shadow=False, fontsize='large')  
    
    plt.grid(axis='y')
    ymin = 0 
    ymax = 7
    ax.set_ylim([0, ymax])    
    ax.set_xlim([1, num_rk])


    plt.xticks(x, labels) #, rotation='vertical')

    return fig
    
    
# -----------------------------------------------------------------------------#
# load data

prefix = "./data/" + run + "/" + run + "_"

t_min_mic = pickle.load(open(prefix+"t_min_mic.p", "rb"))
#t_min_mac = pickle.load(open(prefix+"t_min_mac.p", "rb"))
if berman:
    t_min_mic_ber = pickle.load(open(prefix+"t_min_mic_ber.p", "rb"))
    #t_min_mac_ber = pickle.load(open(prefix+"t_min_mac_ber.p", "rb"))

# -----------------------------------------------------------------------------#
# plot

delta_t = 0.04

if berman:
    fig1 = plot_t_converge_shaded(delta_t, t_min_mic, t_min_mic_ber)
    #fig1 = plot_t_converge_shaded(delta_t, t_min_mic)
    #fig1 = plot_t_converge_shaded(delta_t, t_min_mic_ber)
    #fig2 = plot_t_converge_shaded(delta_t, t_min_mac, t_min_mac_ber)
else:
    fig1 = plot_t_converge_shaded(delta_t, t_min_mic)
plt.axes().set_aspect(0.65,'box')
plt.show()

"""
fig2 = plot_t_converge(delta_t, t_min_mac, t_min_mac_ber)
plt.axes().set_aspect(0.5,'box')
plt.show()
"""

# -----------------------------------------------------------------------------#
# save plots
 
prefix = "./plots/"  + run + "_"

if save_plots:
    fig1.savefig(prefix+'all_rks_t_conv.eps') 
                        



