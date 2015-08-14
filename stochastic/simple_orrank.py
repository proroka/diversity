import itertools
import numpy as np


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
    return int(min_count)


if __name__ == '__main__':
    A = np.random.randint(0, 2, (10, 10))
    print 'OR-rank of\n', A, 'is', orrank(A)
    print 'Matrix-rank is', np.linalg.matrix_rank(A)
    B = np.identity(6)
    print 'OR-rank of\n', B, 'is', orrank(B)
