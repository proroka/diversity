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
run = 'Q28'

plot_orrank = False

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
def plot_t_converge_OR_rk_shaded(delta_t, t_min_a, t_min_b, rks):

    num_graph_iter = t_min_a.shape[0]
    num_traits = t_min_a.shape[1]
    macro = (np.size(t_min_a.shape)==2)
    num_rk = t_min_a.shape[1]
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
    pc_a = np.zeros((num_rk,3))
    pc_b = np.zeros((num_rk,3))
    p0 = 25
    p1 = 50
    p2 = 75
    
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
        pc_a[rk,0] = np.percentile(ta_f[rk],p0)
        pc_a[rk,1] = np.percentile(ta_f[rk],p1)
        pc_a[rk,2] = np.percentile(ta_f[rk],p2)
        pc_b[rk,0] = np.percentile(tb_f[rk],p0)
        pc_b[rk,1] = np.percentile(tb_f[rk],p1)
        pc_b[rk,2] = np.percentile(tb_f[rk],p2)
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
    ymax = 8
    ax.set_ylim([0, ymax])    
    ax.set_xlim([1, num_rk])


    plt.xticks(x, labels) #, rotation='vertical')

    return fig
    
    
# -----------------------------------------------------------------------------#


    
# -----------------------------------------------------------------------------#    
# load data

prefix = "./data/" + run + "/" + run + "_"


list_Q = pickle.load(open(prefix+"list_Q.p", "rb"))
t_min_mic = pickle.load(open(prefix+"t_min_mic.p", "rb"))
#t_min_mac = pickle.load(open(prefix+"t_min_mac.p", "rb"))
#t_min_mic_ber = pickle.load(open(prefix+"t_min_mic_ber.p", "rb"))
#t_min_mac_ber = pickle.load(open(prefix+"t_min_mac_ber.p", "rb"))


# -----------------------------------------------------------------------------#
# get rank

ll = len(list_Q)
rks = np.zeros(ll)

rks_o = np.zeros(ll)
rks_n = np.zeros(ll)

for i in range(ll):
    q = list_Q[i]
    rks_o[i] = orrank(q)
    rks_n[i] = np.linalg.matrix_rank(q)
    if plot_orrank:
        rks[i] = orrank(q)
    else:
        rks[i] = np.linalg.matrix_rank(q)
max_rk = int(max(rks))

# -----------------------------------------------------------------------------#
# plot
delta_t = 0.04


fig1 = plot_t_converge_OR_rk_shaded(delta_t, t_min_mic, t_min_mic, rks)
plt.axes().set_aspect(0.5,'box')
plt.show()

"""
fig2 = plot_t_converge_OR_rk(delta_t, t_min_mac, t_min_mac_ber, rks)
plt.axes().set_aspect(0.5,'box')
plt.show()

"""



