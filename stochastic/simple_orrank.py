import itertools
import numpy as np
import random

# Here is another version that takes O(2^M * N) instead. It also much simpler to understand.
def orrank(A):
    # For all possible combinations of rows check whether we can express all traits.
    B = A.astype(bool)
    all_traits = np.sum(B, axis=0).astype(bool)
    min_count = A.shape[0]
    for rows in itertools.product((False, True), repeat=A.shape[0]):
        row_indices = np.array(rows)
        row_traits = np.sum(B[row_indices], axis=0).astype(bool)
        if np.all(row_traits == all_traits):
            min_count = min(min_count, np.sum(row_indices))
    return min_count


def generate_matrix_with_orrank(nrows, ncols, orrank_value):
    assert orrank_value <= nrows, 'Cannot have OR-rank larger than nrows'
    assert orrank_value <= ncols, 'Cannot have OR-rank larger than ncols'
    # Start with identity matrix.
    A = np.zeros((nrows, ncols))
    np.fill_diagonal(A, 1)
    # Fill random cols.
    if nrows < ncols:
        for i in range(nrows, ncols):
            index = np.random.randint(0, nrows)
            A[index, i] = 1.
    # Fill random rows.
    if nrows > ncols:
        for i in range(ncols, nrows):
            index = np.random.randint(0, ncols)
            A[i, index] = 1.
    # Fill random indices until the or-rank is acheived.
    zeros_indices = np.where(A == 0.)
    zeros_indices = set(zip(zeros_indices[0].tolist(), zeros_indices[1].tolist()))
    while orrank(A) != orrank_value:
        index = random.sample(zeros_indices, 1)[0]
        zeros_indices.remove(index)
        A[index[0], index[1]] = 1.
    return A


if __name__ == '__main__':
    # Little test.
    A = generate_matrix_with_orrank(5, 8, 3)
    print 'OR-rank of\n', A, 'is', orrank(A)
    A = generate_matrix_with_orrank(8, 5, 3)
    print 'OR-rank of\n', A, 'is', orrank(A)
    # Larger test.
    for i in range(3, 10):
        for j in range(3, 10):
            print 'Testing matrix of size %d x %d' % (i, j)
            for r in range(1, min(i, j) + 1):
                assert orrank(generate_matrix_with_orrank(i, j, r)) == r, 'Cannot generate a good matrix of size %d x %d with rank %d' % (i, j, r)
    print 'All good ;)'
