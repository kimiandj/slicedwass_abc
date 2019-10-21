import numpy as np

cpdef swapsweep(int[:] permutation, double[:, :] M, double total_cost):
    cdef int N = M.shape[0]
    cdef double current_cost, proposed_cost
    cdef int perm_i, perm_j
    for i in range(N):
        perm_i = permutation[i]
        for j in range(i+1, N):
            perm_j = permutation[j]
            current_cost = M[i, perm_i] + M[j, perm_j]
            proposed_cost = M[i, perm_j] + M[j, perm_i]
            if proposed_cost < current_cost:
                permutation[i] = perm_j
                permutation[j] = perm_i
                perm_i = perm_j
                total_cost = total_cost - current_cost + proposed_cost
    return [permutation, total_cost]