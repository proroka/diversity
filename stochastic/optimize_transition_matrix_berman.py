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
import warnings

# computes the state after max_time.
# elements is array: flattened k_ij matrix, flattened for all species
def FinalState(elements, adjacency_matrix, max_time, initial_state, transform):
    num_nodes = initial_state.shape[0]
    num_species = initial_state.shape[1]
    deploy_robots_final = np.zeros((num_nodes, num_species))

    K_all = MatrixReshape(elements, adjacency_matrix, num_species)
    # for each species, get transition matrix and calculate final state
    for s in range(num_species):
        K = K_all[:,:,s]
        deploy_robots_final[:,s] = np.dot(scipy.linalg.expm(K * max_time), initial_state[:,s])
    # transform species final state to traits final state
    deploy_traits_final = np.dot(deploy_robots_final, transform)
    return deploy_traits_final


# In:
# desired_state: matrix with node-robot distribution
# elements: vector with transition matrix values for all K_i, speices i
# transform: species-trait matrix
def Cost_Berman(elements, desired_state, adjacency_matrix):
    # Prepare variable depending on whether t part of the parameters.
    num_nodes = adjacency_matrix.shape[0]
    num_species = desired_state.shape[1]
    num_elements_i = np.size(elements) / num_species

    # Reshape adjacency matrix to make sure.
    Adj = adjacency_matrix.astype(float).reshape((num_nodes, num_nodes))
    Adj_flatten = Adj.flatten().astype(bool)  # Flatten boolean version.

    # Loop through the species to compute the cost value.
    cost = 0.0
    for s in range(num_species):
        k_ij = elements[s*num_elements_i:(s+1)*num_elements_i]
        # Create K from individual k_{ij}.
        K = np.zeros(Adj_flatten.shape)
        K[Adj_flatten] = k_ij
        K = K.reshape((num_nodes, num_nodes))
        np.fill_diagonal(K, -np.sum(K, axis=0))
        # Idea: Symmetrize K, find lambda_2.
        xd = desired_state[:, s]
        inv_pi = np.diag(1. / np.sqrt(xd))
        pi = np.diag(np.sqrt(xd))
        Kt = inv_pi.dot(K).dot(pi)
        S = 0.5 * (Kt + Kt.T)
        ws, Vs = scipy.linalg.eig(S, right=True)  # Replace with K to optimize the real K.
        l1 = np.argmax(ws)
        l2 = np.max(np.concatenate([ws[:l1], ws[l1+1:]]))
        cost += l2
    return cost


def Cons_Berman(elements, desired_state, adjacency_matrix):
    # Prepare variable depending on whether t part of the parameters.
    num_nodes = adjacency_matrix.shape[0]
    num_species = desired_state.shape[1]
    num_elements_i = np.size(elements) / num_species

    # Reshape adjacency matrix to make sure.
    Adj = adjacency_matrix.astype(float).reshape((num_nodes, num_nodes))
    Adj_flatten = Adj.flatten().astype(bool)  # Flatten boolean version.

    steadystate = np.empty_like(desired_state)
    for s in range(num_species):
        # Loop through the species to compute the cost value.
        k_ij = elements[s*num_elements_i:(s+1)*num_elements_i]
        # Create K from individual k_{ij}.
        K = np.zeros(Adj_flatten.shape)
        K[Adj_flatten] = k_ij
        K = K.reshape((num_nodes, num_nodes))
        np.fill_diagonal(K, -np.sum(K, axis=0))
        xd = desired_state[:, s]

        wk, Vk = scipy.linalg.eig(K, right=True)
        ss = Vk[:, np.argmax(wk)]
        steadystate[:, s] = ss / np.sum(ss) * np.sum(xd)
    return np.sum(np.square(steadystate - desired_state))


def Cost_Random(elements, desired_state, adjacency_matrix, transform, initial_state):
    # Prepare variable depending on whether t part of the parameters.
    num_nodes = adjacency_matrix.shape[0]
    num_species = initial_state.shape[1]
    num_elements_i = np.size(elements) / num_species

    # Reshape adjacency matrix to make sure.
    Adj = adjacency_matrix.astype(float).reshape((num_nodes, num_nodes))
    Adj_flatten = Adj.flatten().astype(bool)  # Flatten boolean version.

    steadystate = np.empty_like(initial_state)
    for s in range(num_species):
        # Loop through the species to compute the cost value.
        k_ij = elements[s*num_elements_i:(s+1)*num_elements_i]
        # Create K from individual k_{ij}.
        K = np.zeros(Adj_flatten.shape)
        K[Adj_flatten] = k_ij
        K = K.reshape((num_nodes, num_nodes))
        np.fill_diagonal(K, -np.sum(K, axis=0))
        x0 = initial_state[:, s]

        wk, Vk = scipy.linalg.eig(K, right=True)
        ss = Vk[:, np.argmax(wk)]
        steadystate[:, s] = ss / np.sum(ss) * np.sum(x0)
    return np.sum(np.square(steadystate.dot(transform) - desired_state))


# -----------------------------------------------------------------------------#

# reshapes the array of nonzero elements of the matrix into a valid transition matrix.
# returns 3D matrix, 3rd dim indexes the species
def MatrixReshape(elements_array, adjacency_matrix, num_species):
    nstates = adjacency_matrix.shape[0]
    # Place elements where the adjacency matrix has value 1.
    adjacency_matrix = adjacency_matrix.astype(bool)
    a = adjacency_matrix.flatten()
    K_all = np.zeros((nstates, nstates, num_species))
    num_nodes = adjacency_matrix.shape[0]
    num_elements = elements_array.shape[0] / num_species
    for s in range(num_species):
        matrix_elements = np.zeros(nstates ** 2)
        elements = elements_array[s*num_elements:(s+1)*num_elements]
        matrix_elements[a] = elements
        K = np.array(matrix_elements).reshape(num_nodes, num_nodes)
        np.fill_diagonal(K, -np.sum(K, axis=0))
        K_all[:,:,s] = K
    return K_all

# defines bounds on transition matrix values
def ElementsBounds(nelements, max_rate, max_time, f_new, x_new, f_old, x_old):
    if max_time is None:
        elements_min = np.ones(nelements) * 0.  # min_rate.
        elements_max = np.ones(nelements) * max_rate
    else:
        elements_min = np.ones(nelements + 1) * 0.  # min_rate.
        elements_max = np.ones(nelements + 1) * max_rate
        elements_max[-1] = max_time
    return (bool(np.all(x_new <= elements_max)) and
            bool(np.all(x_new >= elements_min)))



# -----------------------------------------------------------------------------#
# optimization: basin hopping

# desired_steadystate is the species desired.
def Optimize_Berman(adjacency_matrix, initial_state, desired_steadystate, transform,
                    max_rate, max_time, verbose):
    global current_iteration
    current_iteration = 0

    # initial array of random elements (only where the adjacency matrix has a 1).
    num_species = desired_steadystate.shape[1]
    num_nonzero_elements = np.sum(adjacency_matrix) # note: diagonal is 0
    # assume that all species have same adjacency matrix
    init_elements = np.random.rand(num_nonzero_elements*num_species) * max_rate
    bounds = [(0., max_rate)] * num_nonzero_elements*num_species

    CostFunction = lambda x: Cost_Berman(x, desired_steadystate, adjacency_matrix)
    BoundFunction = lambda f_new, x_new, f_old, x_old: ElementsBounds(num_nonzero_elements*num_species, max_rate, None, f_new, x_new, f_old, x_old)

    #---
    # basinhopping function
    minimizer_kwargs = {'constraints': {'type': 'eq', 'fun': lambda x: Cons_Berman(x, desired_steadystate, adjacency_matrix)},
                        'bounds': bounds, 'options': {'disp': True}}
    success = False
    while not success:
        try:
            ret = scipy.optimize.basinhopping(CostFunction,
                                              init_elements,
                                              minimizer_kwargs=minimizer_kwargs,
                                              niter=100, niter_success=3,  # Actually since it is convex, we're good with one iteration.
                                              accept_test=BoundFunction)
            # success = True
            success = (ret.fun < 0.0)
        except (ValueError, np.linalg.linalg.LinAlgError) as e:
            print 'Problem during optimization:', e, '- Retrying...'
            # Make completely new random elements.
            init_elements = np.random.rand(num_nonzero_elements*num_species) * max_rate
            success = False


    if verbose:
        print '\n\nFinal matrix:\n', MatrixReshape(ret.x, adjacency_matrix, num_species)
        print '\nSum columns:', np.sum(MatrixReshape(ret.x, adjacency_matrix, num_species), axis=0)
        print '\nReaches:\n', FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform)
        print '\nError (at time %.2f):\n' % max_time, FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform) - desired_steadystate.dot(transform)

    # Print this always
    print '\nFinal cost:\n', ret.fun

    # return transition matrices (3D matrix for all species)
    return MatrixReshape(ret.x, adjacency_matrix, num_species)



# desired_steadystate is the trait desired.
def Optimize_Random(adjacency_matrix, initial_state, desired_steadystate, transform,
                    max_rate, max_time, verbose):
    global current_iteration
    current_iteration = 0

    # initial array of random elements (only where the adjacency matrix has a 1).
    num_species = initial_state.shape[1]
    num_nonzero_elements = np.sum(adjacency_matrix) # note: diagonal is 0
    # assume that all species have same adjacency matrix
    init_elements = np.random.rand(num_nonzero_elements*num_species) * max_rate
    bounds = [(0., max_rate)] * num_nonzero_elements*num_species

    CostFunction = lambda x: Cost_Random(x, desired_steadystate, adjacency_matrix, transform, initial_state)
    BoundFunction = lambda f_new, x_new, f_old, x_old: ElementsBounds(num_nonzero_elements*num_species, max_rate, None, f_new, x_new, f_old, x_old)

    #---
    # basinhopping function
    minimizer_kwargs = {'bounds': bounds, 'options': {'disp': True}}
    success = False
    while not success:
        try:
            ret = scipy.optimize.basinhopping(CostFunction,
                                              init_elements,
                                              minimizer_kwargs=minimizer_kwargs,
                                              niter=1000,
                                              accept_test=BoundFunction)
            success = True
        except (ValueError, np.linalg.linalg.LinAlgError) as e:
            print 'Problem during optimization:', e, '- Retrying...'
            # Make completely new random elements.
            init_elements = np.random.rand(num_nonzero_elements*num_species) * max_rate
            success = False


    if verbose:
        print '\n\nFinal matrix:\n', MatrixReshape(ret.x, adjacency_matrix, num_species)
        print '\nSum columns:', np.sum(MatrixReshape(ret.x, adjacency_matrix, num_species), axis=0)
        print '\nReaches:\n', FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform)
        print '\nError (at time %.2f):\n' % max_time, FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform) - desired_steadystate

    # Print this always
    print '\nFinal cost:\n', ret.fun

    # return transition matrices (3D matrix for all species)
    return MatrixReshape(ret.x, adjacency_matrix, num_species)




