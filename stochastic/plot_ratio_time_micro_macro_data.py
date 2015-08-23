# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:05:33 2015

@author: amandaprorok
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx
import sys
import pickle
import time
from funcdef_util_heterogeneous import *

run = 'V20'
match = 1 # 0 for V21, 1 for V20
delta_t = 0.04

prefix = "./data/" + run + "/" + run + "_micmac_"


    
species_traits = pickle.load(open(prefix+"st.p", "rb"))
deploy_traits_desired = pickle.load(open(prefix+"dtd.p", "rb"))
deploy_robots_micro = pickle.load(open(prefix+"drm.p", "rb"))
deploy_robots_euler = pickle.load(open(prefix+"dre.p", "rb"))
    
    
fig = plot_traits_ratio_time_micmac(deploy_robots_micro, deploy_robots_euler, deploy_traits_desired, 
                              species_traits, delta_t, match)
plt.axes().set_aspect(8/.4,'box') # for V20
#plt.axes().set_aspect(8/.16,'box') # for V21
plt.show()


fig.savefig('./plots/' + run + '_trait_time_micmac.eps')     

                              