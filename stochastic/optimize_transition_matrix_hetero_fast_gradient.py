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

# -----------------------------------------------------------------------------#
# helper functions to verify the gradient.

# Little helper to make nice looking terminal prints.
def hilite(string, success, failure, bold):
    attr = []
    if success:
        attr.append('32')  # green
    if not success and failure:
        attr.append('31')  # red
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def gradcheck_naive(f, x):
    fx, grad = f(x)  # Evaluate function value at original point
    assert x.shape == grad.shape, 'Variable and gradient must have the same shape'
    h = 1e-4
    passed = True
    numerical_grad = np.empty_like(x)

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        x[ix] -= h
        y1 = f(x)[0]
        x[ix] += 2 * h
        y2 = f(x)[0]
        numgrad = (y2 - y1) / (2 * h)
        x[ix] -= h
        numerical_grad[ix] = numgrad
        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5 and passed:
            # Only print the first error.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.ComplexWarning)  # Ignore complex warning.
                print hilite("Gradient check failed.", False, True, True)
                print "First gradient error found at index %s" % str(ix)
                print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            passed = False
        it.iternext()  # Step to next dimension
    if passed:
        print hilite("Gradient check passed!", True, False, True)
    return numerical_grad


# -----------------------------------------------------------------------------#
# auxiliary functions


# to give feedback during the optimization.
current_iteration = 0
def Print(x, f, accepted):
    global current_iteration
    current_iteration += 1
    sys.stdout.write('\r%i) Cost: %.4f, Accepted %s' % (current_iteration, f, str(accepted)))
    sys.stdout.flush()

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

# -----------------------------------------------------------------------------#

# defines the cost to minimize.
# parameter elements is array - concatenation of nonzero elements for each species
def Cost_OLD(elements, desired_state, adjacency_matrix, max_time, initial_state, transform):
    #return np.mean(np.square(desired_state - FinalState(elements, adjacency_matrix, max_time, initial_state, transform))
    diffsq = np.square(desired_state - FinalState(elements, adjacency_matrix, max_time, initial_state, transform))
    return np.sum(np.sum(diffsq))

# -----------------------------------------------------------------------------#

# Different optimization modes.
ABSOLUTE_AT_LEAST = 0
ABSOLUTE_EXACT = 1
QUADRATIC_AT_LEAST = 2
QUADRATIC_EXACT = 3

# Computes V * exp_wt * U.
# By construction the exponential of our matrices are always real-valued.
def Expm(V, exp_wt, U):
    return np.real(V.dot(np.diag(exp_wt)).dot(U))

# In:
# desired_state: matrix with node-trait distribution
# elements: vector with transition matrix values for all K_i, speices i
# transform: species-trait matrix
# initial_state: node-species matrix
#
# If max_time is None, the optimization assumes that the time is the last element
# in elements and the function will compute the gradient w.r.t. to t too.
#
# If alpha > 0, the cost also minimizes (alpha * t^2).
#
# If steady_state_dt > 0 and beta > 0, the optimization will check the final state at t and t + dt.
def Cost_Fast(elements, desired_state, adjacency_matrix, max_time, initial_state, transform,
              cost_mode=QUADRATIC_EXACT, alpha=0.0, margin=0.0, steady_state_dt=0.0, beta=5.0):
    # Prepare variable depending on whether t part of the parameters.
    num_nodes = adjacency_matrix.shape[0]
    num_species = initial_state.shape[1]
    num_traits = transform.shape[1]
    if max_time is None:
        t = elements[-1]
        num_elements_i = (np.size(elements) - 1) / num_species
        grad_all = np.zeros(np.size(elements))
        all_elements = elements[:-1]
    else:
        alpha = 0.0  # Overwrite alpha if we are not optimizing t.
        t = max_time
        num_elements_i = np.size(elements) / num_species
        grad_all = np.zeros(np.size(elements))
        all_elements = elements

    # Reshape adjacency matrix to make sure.
    Adj = adjacency_matrix.astype(float).reshape((num_nodes, num_nodes))
    Adj_flatten = Adj.flatten().astype(bool)  # Flatten boolean version.

    # Loop through the species to compute the cost value.
    # At the same time, prepare the different matrices.
    Ks = []                     # K_s
    eigenvalues = []            # w
    eigenvectors = []           # V.T
    eigenvectors_inverse = []   # U.T
    exponential_wt = []         # exp(eigenvalues * t).
    x_matrix = []               # Pre-computed X matrices.
    x0s = []                    # Avoids reshaping.
    qs = []                     # Avoids reshaping.
    xts = []                    # Keeps x_s(t).
    inside_norm = np.zeros((num_nodes, num_traits))  # Will hold the value prior to using the norm.
    for s in range(num_species):
        x0 = initial_state[:,s].reshape((num_nodes, 1))
        q = transform[s,:].reshape((1, num_traits))
        x0s.append(x0)
        qs.append(q)
        k_ij = elements[s*num_elements_i:(s+1)*num_elements_i]
        # Create K from individual k_{ij}.
        K = np.zeros(Adj_flatten.shape)
        K[Adj_flatten] = k_ij
        K = K.reshape((num_nodes, num_nodes))
        np.fill_diagonal(K, -np.sum(K, axis=0))
        # Store K.
        Ks.append(K)
        # Perform eigen-decomposition to compute matrix exponential.
        w, V = scipy.linalg.eig(K, right=True)
        U = scipy.linalg.inv(V)
        wt = w * t
        exp_wt = np.exp(wt)
        xt = Expm(V, exp_wt, U).dot(x0)
        inside_norm += xt.dot(q)
        # Store the transpose of these matrices for later use.
        eigenvalues.append(w)
        eigenvectors.append(V.T)
        eigenvectors_inverse.append(U.T)
        exponential_wt.append(exp_wt)
        xts.append(xt)
        # Pre-build X matrix.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0 on the diagonal.
            X = np.subtract.outer(exp_wt, exp_wt) / (np.subtract.outer(wt, wt) + 1e-10)
        np.fill_diagonal(X, exp_wt)
        x_matrix.append(X)
    inside_norm -= desired_state

    # Compute the final cost value depending on cost_mode.
    derivative_outer_norm = None  # Holds the derivative of inside_norm (except the multiplication by (x0 * q)^T).
    if cost_mode == ABSOLUTE_AT_LEAST:
        derivative_outer_norm = -inside_norm + margin
        value = np.sum(np.maximum(derivative_outer_norm, 0))
        derivative_outer_norm = -(derivative_outer_norm > 0).astype(float)  # Keep only 1s for when it's larger than margin.
    elif cost_mode == ABSOLUTE_EXACT:
        abs_inside_norm = np.abs(inside_norm)
        index_zeros = abs_inside_norm < 1e-10
        value = np.sum(np.abs(inside_norm))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0.
            derivative_outer_norm = inside_norm / abs_inside_norm # Keep only 1s for when it's larger than 0 and -1s for when it's lower.
        derivative_outer_norm[index_zeros] = 0  # Make sure we set 0/0 to 0.
    elif cost_mode == QUADRATIC_AT_LEAST:
        derivative_outer_norm = -inside_norm + margin
        value = np.sum(np.square(np.maximum(derivative_outer_norm, 0)))
        index_negatives = derivative_outer_norm < 0
        derivative_outer_norm *= -2.0
        derivative_outer_norm[index_negatives] = 0  # Don't propagate gradient on negative values.
    elif cost_mode == QUADRATIC_EXACT:
        value = np.sum(np.square(inside_norm))
        derivative_outer_norm = 2.0 * inside_norm
    value += alpha * (t ** 2)

    # Calculate gradient w.r.t. the transition matrix for each species
    for s in range(num_species):
        # Build gradient w.r.t. inside_norm of cost.
        top_grad = np.dot(derivative_outer_norm, np.dot(x0s[s], qs[s]).T)
        # Build gradient w.r.t. Exp(K * t).
        middle_grad = eigenvectors_inverse[s].dot(eigenvectors[s].dot(top_grad).dot(eigenvectors_inverse[s]) * x_matrix[s]).dot(eigenvectors[s])
        # Build gradient w.r.t. K
        bottom_grad = middle_grad * t
        # Finally, propagate gradient to individual k_ij.
        grad = bottom_grad - np.diag(bottom_grad)
        grad = grad.flatten()[Adj_flatten]  # Reshape.
        grad_all[s*num_elements_i:(s+1)*num_elements_i] += np.array(np.real(grad))
        # Build gradient w.r.t. t (if desired)
        if max_time is None:
            grad_all[-1] += np.real(np.sum(Ks[s] * middle_grad))

    # Gradient of alpha * t^2 w.r.t. t
    if max_time is None:
        grad_all[-1] += 2.0 * t * alpha

    # Forcing the steady state.
    # We add a cost for keeping x(t) and x(t + dt) the same. We use the quadratic norm for sub-cost.
    # The larger beta and the larger steady_state_dt, the more equal these state will be.
    if steady_state_dt and beta:
        for s in range(num_species):
            # Compute exp of the eigenvalues of K * (t + dt).
            wtdt = eigenvalues[s] * (t + steady_state_dt)
            exp_wtdt = np.exp(wtdt)
            # Compute x_s(t) - x_s(t + dt) for that species.
            # Note that since we store V.T and U.T, we do (U.T * D * V.T).T == V * D * U
            inside_norm = xts[s] - Expm(eigenvectors_inverse[s], exp_wtdt, eigenvectors[s]).T.dot(x0s[s])
            # Increment value.
            value += beta * np.sum(np.square(inside_norm))

            # Compute gradient on the first part of the cost: e^{Kt} x0 (we use the same chain rule as before).
            top_grad = 2.0 * beta * np.dot(inside_norm, x0s[s].T)
            store_inner_product = eigenvectors[s].dot(top_grad).dot(eigenvectors_inverse[s])  # Store to re-use.
            middle_grad = eigenvectors_inverse[s].dot(store_inner_product * x_matrix[s]).dot(eigenvectors[s])
            bottom_grad = middle_grad * t
            grad = bottom_grad - np.diag(bottom_grad)
            grad = grad.flatten()[Adj_flatten]  # Reshape.
            grad_all[s*num_elements_i:(s+1)*num_elements_i] += np.array(np.real(grad))
            if max_time is None:
                grad_all[-1] += np.real(np.sum(Ks[s] * middle_grad))

            # Compute gradient on the second part of the cost: e^{K(t + dt)} x0 (we use the same chain rule as before).
            # Compute X for e^{K(t + dt)}.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0 on the diagonal.
                X = np.subtract.outer(exp_wtdt, exp_wtdt) / (np.subtract.outer(wtdt, wtdt) + 1e-10)
            np.fill_diagonal(X, exp_wtdt)
            # top_grad = 2.0 * beta * np.dot(inside_norm, x0s[s].T) [same as before but needs to be negated].
            middle_grad = -eigenvectors_inverse[s].dot(store_inner_product * X).dot(eigenvectors[s])
            bottom_grad = middle_grad * (t + steady_state_dt)
            grad = bottom_grad - np.diag(bottom_grad)
            grad = grad.flatten()[Adj_flatten]  # Reshape.
            grad_all[s*num_elements_i:(s+1)*num_elements_i] += np.array(np.real(grad))
            if max_time is None:
                grad_all[-1] += np.real(np.sum(Ks[s] * middle_grad))

    return [value, grad_all]



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

def Optimize_Hetero_Fast(init_values, adjacency_matrix, initial_state, desired_steadystate,
                         transform, max_time, max_rate, verbose, l_norm, match,
                         optimizing_t, force_steady_state):
    global current_iteration
    current_iteration = 0

    if l_norm==1 and match==0:
        cost_mode = ABSOLUTE_AT_LEAST
    if l_norm==1 and match==1:
        cost_mode = ABSOLUTE_EXACT
    if l_norm==2 and match==0:
        cost_mode = QUADRATIC_AT_LEAST
    if l_norm==2 and match==1:
        cost_mode = QUADRATIC_EXACT

    # settings
    verify_gradient = False
    alpha = 1.0 #0.1  # + alpha * t^2.
    mu = 0.0  # margin

    # initial array of random elements (only where the adjacency matrix has a 1).
    num_species = initial_state.shape[1]
    num_nonzero_elements = np.sum(adjacency_matrix) # note: diagonal is 0
    if init_values.shape[0]==0:
        # assume that all species have same adjacency matrix
        init_elements = np.random.rand(num_nonzero_elements*num_species) * max_rate
    else:

        init_elements = np.zeros(num_nonzero_elements*num_species)
        adjacency_matrix = adjacency_matrix.astype(bool)
        a = adjacency_matrix.flatten()
        for s in range(num_species):
            temp = init_values[:,:,s]
            init_elements[s*num_nonzero_elements:(s+1)*num_nonzero_elements] = temp[adjacency_matrix].flatten()


    bounds = [(0., max_rate)] * num_nonzero_elements*num_species


    # Check gradient if requested.
    if verify_gradient:
        init_elements_with_time = np.concatenate([init_elements, np.array([max_time])], axis=0)
        # Randomly modify the init_elements and verify gradient 10 times.
        for i in range(10):
            new_init_elements = np.random.rand(*init_elements.shape)
            new_init_elements_with_time = np.random.rand(*init_elements_with_time.shape)
            margin = np.random.rand(1)
            optimize_t = np.random.rand(1)
            for mode in [ABSOLUTE_AT_LEAST, ABSOLUTE_EXACT, QUADRATIC_AT_LEAST, QUADRATIC_EXACT]:
                # Without optimizing t.
                gradcheck_naive(lambda x: Cost_Fast(x, desired_steadystate, adjacency_matrix, max_time, initial_state, transform,
                                                    cost_mode=mode, margin=margin), new_init_elements)
                # Optimizing t.
                gradcheck_naive(lambda x: Cost_Fast(x, desired_steadystate, adjacency_matrix, None, initial_state, transform,
                                                    cost_mode=mode, margin=margin, optimize_t=optimize_t, steady_state_dt=force_steady_state), new_init_elements_with_time)

    # Prepare cost function.
    if optimizing_t:
        init_elements = np.concatenate([init_elements, np.array([max_time])], axis=0)
        bounds.append((0., max_time * 2))  # Allow optimized time be up to twice as large as max_time.
        CostFunction = lambda x: Cost_Fast(x, desired_steadystate, adjacency_matrix, None, initial_state, transform, cost_mode=cost_mode, alpha=alpha, margin=mu, steady_state_dt=force_steady_state)
        BoundFunction = lambda f_new, x_new, f_old, x_old: ElementsBounds(num_nonzero_elements*num_species, max_rate, 2.0 * max_time, f_new, x_new, f_old, x_old)
    else:
        CostFunction = lambda x: Cost_Fast(x, desired_steadystate, adjacency_matrix, max_time, initial_state, transform, cost_mode=cost_mode, margin=mu)
        BoundFunction = lambda f_new, x_new, f_old, x_old: ElementsBounds(num_nonzero_elements*num_species, max_rate, None, f_new, x_new, f_old, x_old)


    #---
    # basinhopping function
    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'jac': True, 'options': {'disp': False}}
    success = False
    while not success:
        # It happens very rarely that the eigenvector matrix becomes close to singular and
        # cannot be inverted. In that case, we simply restart the optimization.
        try:
            ret = scipy.optimize.basinhopping(CostFunction,
                                              init_elements,
                                              minimizer_kwargs=minimizer_kwargs,
                                              niter=100, niter_success=3,
                                              accept_test=BoundFunction,
                                              callback=None if not verbose else Print)
            # success = True
            success = (ret.fun < 1e3)
        except (ValueError, np.linalg.linalg.LinAlgError) as e:
            print 'Problem during optimization:', e, '- Retrying...'
            # Make completely new random elements.
            init_elements = np.random.rand(num_nonzero_elements*num_species) * max_rate
            if optimizing_t:
                init_elements = np.concatenate([init_elements, np.array([max_time])], axis=0)
            success = False

    # Remove the optimized t.
    if optimizing_t:
        optimal_t = ret.x[-1]
        ret.x = ret.x[:-1]
    else:
        optimal_t = max_time

    if verbose:
        print '\n\nFinal matrix:\n', MatrixReshape(ret.x, adjacency_matrix, num_species)
        print '\nSum columns:', np.sum(MatrixReshape(ret.x, adjacency_matrix, num_species), axis=0)
        print '\nReaches:\n', FinalState(ret.x, adjacency_matrix, optimal_t, initial_state, transform)
        print '\nError (at time %.2f):\n' % optimal_t, FinalState(ret.x, adjacency_matrix, optimal_t, initial_state, transform) - desired_steadystate
        if optimizing_t:
            print '\nOptimal time:', optimal_t
            print '\nError (at max time %.2f):\n' % max_time, FinalState(ret.x, adjacency_matrix, max_time, initial_state, transform) - desired_steadystate


    # Print this always
    print '\nFinal cost:\n', ret.fun

    # return transition matrices (3D matrix for all species)
    return MatrixReshape(ret.x, adjacency_matrix, num_species)









