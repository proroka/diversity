# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:17:31 2015
@author: amandaprorok

"""


import numpy as np
import pylab as pl
import matplotlib as plt
import networkx as nx


# -----------------------------------------------------------------------------#
# functional diversity, Petchey 2002

def show_graph(adjacency_matrix):
    # given an adjacency matrix use networkx and matplotlib to plot graph
    

    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    # nx.draw(gr) # edited to include labels
    #nx.draw_networkx(gr)
    nx.draw(gr)
    # now if you decide you don't want labels because your graph
    # is too busy just do: nx.draw_networkx(G,with_labels=False)
    #plt.show() 