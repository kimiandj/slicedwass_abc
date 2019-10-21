import numpy as np

cdef extern from "HilbertCode.cpp":
        void Hilbert_Sort_CGAL(double *x, int dx, int N, int *J)

cpdef hilbert_order_(double[:, :] x):
        cdef int dx, N
        cdef int[:] y
        dx = x.shape[0]
        N = x.shape[1]
        y = np.zeros(N, dtype=np.int32)
        Hilbert_Sort_CGAL(&x[0, 0], dx, N, &y[0])
        return np.asarray(y)
