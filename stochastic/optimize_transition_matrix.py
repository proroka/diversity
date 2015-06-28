# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:39:47 2015
@author: amandaprorok

"""

import scipy
import scipy.optimize
import scipy.linalg
import numpy as np
import sys


# -----------------------------------------------------------------------------#
# auxiliary functions
    

# To give feedback during the optimization.
def Print(x, f, accepted, current_iteration):
    current_iteration += 1
    sys.stdout.write('\r%i) Cost: %.4f, Accepted %s' % (current_iteration, f, str(accepted)))
    sys.stdout.flush()

# Computes the state after max_time.
def FinalState(elements, adjacency_matrix, max_time, initial_state):
    return np.dot(scipy.linalg.expm(MatrixReshape(elements, adjacency_matrix) * max_time),initial_state)

# Defines the cost to minimize.
def Cost(elements, desired_state, adjacency_matrix, max_time, initial_state):
    return np.mean(np.square(desired_state - FinalState(elements, adjacency_matrix, max_time, initial_state)))


# Reshapes the elements of the matrix into a valid transition matrix.
# Size of elements is nstates ** 2 - nstates.
def MatrixReshape(elements, adjacency_matrix):
    nstates = adjacency_matrix.shape[0]
    matrix_elements = np.zeros(nstates ** 2)
    # Place elements where the adjacency matrix has value 1.
    adjacency_matrix = adjacency_matrix.astype(bool)
    a = adjacency_matrix.flatten()
    matrix_elements[a] = elements
    K = np.array(matrix_elements).reshape(nstates, nstates)
    np.fill_diagonal(K, -np.sum(K, axis=0))
    return K
    
def ElementBounds(nelements, max_rate):
    elements_min = np.ones(nelements) * 0.  # min_rate.
    elements_max = np.ones(nelements) * max_rate
    
    def __call__(self, f_new, x_new, f_old, x_old):
        return (bool(np.all(x_new <= self.elements_max)) and bool(np.all(x_new >= self.elements_min)))


              
# -----------------------------------------------------------------------------#
# optimization

def Optimize(adjacency_matrix, initial_state, desired_steadystate, max_time, max_rate, verbose):
    current_iteration = 0

    # Initial random elements (only where the adjacency matrix has a 1).
    num_offdiag_elements = np.sum(adjacency_matrix)
    init_elements = np.random.rand(num_offdiag_elements) * max_rate
    
    bounds = [(0., max_rate)] * num_offdiag_elements
    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds} # L-BFGS-B is the one which takes bounds
    
    # basinhop optimization
    ret = scipy.optimize.basinhopping(lambda x: Cost(x, desired_steadystate, adjacency_matrix, max_time, initial_state), 
                                      init_elements, 
                                      minimizer_kwargs=minimizer_kwargs, 
                                      niter=100,
                                      accept_test=ElementBounds(num_offdiag_elements, max_rate),
                                      callback=None if not verbose else lambda x, f, a: Print(x, f, a, current_iteration))
    
    if verbose:
        print '\n\nFinal matrix:\n', MatrixReshape(ret.x, adjacency_matrix)
        print '\nSum columns:', np.sum(MatrixReshape(ret.x, adjacency_matrix), axis=0)
        print '\nFinal cost:\n', ret.fun
        print '\nReaches:\n', FinalState(ret.x, adjacency_matrix, max_time, initial_state)
        print '\nError:\n', FinalState(ret.x, adjacency_matrix, max_time, initial_state) - desired_steadystate
        
    return MatrixReshape(ret.x, adjacency_matrix)
      
      