# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:12:53 2016

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
sys.path.append('../plotting')
sys.path.append('../utilities')
sys.path.append('..')
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod

# -----------------------------------------------------------------------------#
# utilities

# returns time of success; if no success, return num_time_steps
def get_convergence_time(ratio, min_ratio):
    for t in range(len(ratio)):     
        if ratio[t] <= min_ratio:
            return t
    return t

# defines relationship between alpha and beta        
def relation_ab(b, range_alpha):
    a = len(range_alpha) - b - 1
    return a