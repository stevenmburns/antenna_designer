import numpy as np

from .abstract_pysim import AbstractPySim

from .pysim_accelerators import psi_fusion_trapezoid
from icecream import ic

def Integral_Standalone(m, n, *, ntrap, wire_radius, k):
    m_centers = m[1:-1:2,:]
    n_endpoints = n[::2,:]

    vec_delta = n_endpoints[1:,:] - n_endpoints[:-1,:]
    delta = np.sqrt((vec_delta*vec_delta).sum(axis=1))
    assert n_endpoints.shape[0] - 1 == delta.shape[0]

    def Aux(theta):
        local_n = n_endpoints[:-1,:]*(1-theta) + theta*n_endpoints[1:,:]

        diffs = local_n[np.newaxis, :, :] - m_centers[:, np.newaxis, :]
        R = np.sqrt((diffs*diffs).sum(axis=2))

        # not always diagonal indices
        diag_indices = np.where(R < 0.00001)
        new_delta = delta[diag_indices[0]]

        RR = R
        RR[diag_indices] = 1

        local_res = np.exp(-(0+1j)*k*R)/(4*np.pi*RR)
        diag = 1/(2*np.pi*new_delta) * np.log(new_delta/wire_radius) - (0+1j)*k/(4*np.pi) 
        local_res[diag_indices] = diag

        return local_res

    res = np.zeros(shape=(m_centers.shape[0], n_endpoints.shape[0]-1),dtype=np.complex128)
    if ntrap == 0:
        res += Aux(0.5)
    else:
        for i in range(0, ntrap+1):
            theta = i/ntrap
            coeff = (2 if i > 0 and i < ntrap else 1)/(2*ntrap)
            res += coeff * Aux(theta)

    return res

class PySim(AbstractPySim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_impedance(self, *, ntrap=0, engine='accelerated'):

        y0, y1 = np.float64(0), np.float64(2*self.halfdriver)

        p0, p1 = np.array((0, y0, 0),dtype=np.float64), np.array((0, y1, 0),dtype=np.float64)

        delta_p = (p1-p0)/(2*self.nsegs)
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
        exnm = np.linspace(p0-delta_p, p1+delta_p, 2*self.nsegs+3)

        a_pts     = exnm[1:-1,:]
        assert a_pts.shape == (2*self.nsegs+1,3)

        vec_delta_l = exnm[1:-3:2,:] - exnm[3:-1:2,:]
        assert vec_delta_l.shape == (self.nsegs,3)


        def Integral_Test(n, m, ntrap):
            res_python = Integral_Python(n, m, ntrap=ntrap)
            res_accelerated = Integral_Accelerated(n, m, ntrap=ntrap)
            assert (abs(res_python-res_accelerated) < 0.001).all()
            return res_accelerated

        def Integral_Python(n, m, ntrap):
            return Integral_Standalone(n, m, ntrap=ntrap, wire_radius=self.wire_radius, k=self.k)

        def Integral_Accelerated(n, m, ntrap):
            return psi_fusion_trapezoid(n, m, wire_radius=self.wire_radius, k=self.k, ntrap=ntrap)

        if engine == 'accelerated':
            Integral = Integral_Accelerated
        elif engine == 'python':
            Integral = Integral_Python
        elif engine == 'test':
            Integral = Integral_Test
        else:
            assert False # pragma: no cover

        z = self.jomega * self.mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)

        z *= Integral(a_pts, a_pts, ntrap=ntrap)

        s = 1/(self.jomega*self.eps) * Integral(exnm, exnm, ntrap=ntrap)

        z += s[:-1,:-1] + s[1:, 1:] - s[:-1, 1:] - s[1:, :-1]
        
        self.z = z

        return self.factor_and_solve()

