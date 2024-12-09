# cython: language_level=3

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "<complex.h>":
     double complex cexp(double complex)

DTYPE = np.int64
DTYPEFLOAT = np.float64
DTYPECOMPLEX = np.complex128

ctypedef long DTYPE_t
ctypedef double DTYPEFLOAT_t
ctypedef double complex DTYPECOMPLEX_t

def cython_compute_impedance_matrix(*,
        DTYPEFLOAT_t halfdriver,
        DTYPE_t nsegs,
        DTYPEFLOAT_t k,
        DTYPEFLOAT_t wire_radius,
        DTYPECOMPLEX_t jomega,
        DTYPEFLOAT_t mu,
        DTYPEFLOAT_t eps):

    cdef DTYPEFLOAT_t y0 = 0
    cdef DTYPEFLOAT_t y1 = 2*halfdriver

    cdef np.ndarray[DTYPEFLOAT_t, ndim=1] p0 = np.array((0, y0, 0))
    cdef np.ndarray[DTYPEFLOAT_t, ndim=1] p1 = np.array((0, y1, 0))

    cdef np.ndarray[DTYPEFLOAT_t, ndim=1] delta_p = (p1-p0)/(2*nsegs)

    """
    exnm - extended nodes and midpoints, there is a point on either end so we can use it to compute delta_l on the boundaries
    for a wire with nseg=3 segments extending 0 to 3 there are three wires:
         [0, 1], [1, 2], [2, 3]
    the exnm array would halve extra points on the boundaries and at the midpoints

    -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5

      0   1   2   3   4   5   6   7   8

    There are 2*nseg + 3 points, nseg of the midpoints, nseg + 1 for the wire endpoints,
    and 2 more the points outside the boundary

    delta_l is the length of each segment.
    You can find this subtract adjacent elements in the subarray with indices [1,3,5,7]
    --- delta_l_plus, its [2,4,6,8], and delta_l_minus, its [0,2,4,6].

    The points themselves are at indices: [2, 4, 6],
    minus at [1, 3, 5] and plus at [3, 5, 7]
    """
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] exnm = np.empty(shape=(2*nsegs+3, 3), dtype=DTYPEFLOAT)

    exnm[0] = p0-delta_p

    cdef i
    for i in range(1,2*nsegs+3):
        exnm[i] = exnm[i-1] + delta_p

    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] vec_delta_l_minus = exnm[ :-4:2,:] - exnm[2:-2:2,:]
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] vec_delta_l       = exnm[1:-3:2,:] - exnm[3:-1:2,:]
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] vec_delta_l_plus  = exnm[2:-2:2,:] - exnm[4:  :2,:]


    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] pts_minus = exnm[1:-3:2,:]
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] pts       = exnm[2:-2:2,:]
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] pts_plus  = exnm[3:-1:2,:]


    cdef np.ndarray[DTYPEFLOAT_t, ndim=1] delta_l = np.sqrt((vec_delta_l**2).sum(axis=1))
    cdef np.ndarray[DTYPEFLOAT_t, ndim=1] delta_l_plus = np.sqrt((vec_delta_l_plus**2).sum(axis=1))
    cdef np.ndarray[DTYPEFLOAT_t, ndim=1] delta_l_minus = np.sqrt((vec_delta_l_minus**2).sum(axis=1))

    def Integral(np.ndarray[DTYPEFLOAT_t, ndim=2] n, np.ndarray[DTYPEFLOAT_t, ndim=2] m, np.ndarray[DTYPEFLOAT_t, ndim=1] delta):

        assert n.shape[0] == delta.shape[0]

        cdef np.ndarray[DTYPEFLOAT_t, ndim=3] diffs = n[np.newaxis, :, :] - m[:, np.newaxis, :]
        cdef np.ndarray[DTYPEFLOAT_t, ndim=2] R = np.sqrt((diffs*diffs).sum(axis=2))

        # not always diagonal indices
        cdef tuple diag_indices = np.where(R == 0)
        cdef np.ndarray[DTYPE_t, ndim=1] row_indices = diag_indices[0]
        cdef np.ndarray[DTYPE_t, ndim=1] col_indices = diag_indices[1]

        cdef np.ndarray[DTYPEFLOAT_t, ndim=1] new_delta = delta[diag_indices[0]]

        cdef np.ndarray[DTYPEFLOAT_t, ndim=2] RR = R

        cdef DTYPE_t i
        for i in range(row_indices.shape[0]):
            RR[row_indices[i], col_indices[i]] = 1

        cdef np.ndarray[DTYPECOMPLEX_t, ndim=2] res = np.exp(-(0+1j)*k*R)/(4*np.pi*RR)

        cdef np.ndarray[DTYPECOMPLEX_t, ndim=1] diag = 1/(2*np.pi*new_delta) * np.log(new_delta/wire_radius) - (0+1j)*k/(4*np.pi) 

        for i in range(row_indices.shape[0]):
            res[row_indices[i],col_indices[i]] = diag[i]

        return res

    cdef np.ndarray[DTYPECOMPLEX_t, ndim=2] z = jomega * mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)

    z *= Integral(pts, pts, delta_l)

    z += 1/(jomega*eps) * Integral(pts_plus, pts_plus, delta_l_plus)
    z -= 1/(jomega*eps) * Integral(pts_plus, pts_minus, delta_l_plus)
    z -= 1/(jomega*eps) * Integral(pts_minus, pts_plus, delta_l_minus)
    z += 1/(jomega*eps) * Integral(pts_minus, pts_minus, delta_l_minus)

    return z
