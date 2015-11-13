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