import numpy as np

from .abstract_pysim import AbstractPySim

from .pysim_accelerators import new_psi_fusion_trapezoid
from icecream import ic

def Integral_Standalone(l_endpoints, r_endpoints, *, ntrap, wire_radius, k):
    m_centers = (l_endpoints + r_endpoints) / 2

    vec_delta = r_endpoints - l_endpoints
    delta = np.sqrt((vec_delta*vec_delta).sum(axis=1))
    assert m_centers.shape[0] == delta.shape[0]

    def Aux(theta):
        local_n = l_endpoints*(1-theta) + theta*r_endpoints

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

    res = np.zeros(shape=(m_centers.shape[0], m_centers.shape[0]),dtype=np.complex128)
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

    def compute_impedance(self, *, ntrap=0, engine='python'):

        y = np.float64(self.halfdriver)

        p0, p1 = np.array((0, -y, 0),dtype=np.float64), np.array((0, y, 0),dtype=np.float64)

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

        def Integral_Python(l_endpoints, r_endpoints, ntrap):
            return Integral_Standalone(l_endpoints, r_endpoints, ntrap=ntrap, wire_radius=self.wire_radius, k=self.k)

        def Integral_Accelerated(n, m, ntrap):
            return new_psi_fusion_trapezoid(l_endpoints, r_endpoints, wire_radius=self.wire_radius, k=self.k, ntrap=ntrap)

        def Integral_Test(l_endpoints, r_endpoints, ntrap):
            res_python = Integral_Python(l_endpoints, r_endpoints, ntrap=ntrap)
            res_accelerated = Integral_Accelerated(l_endpoints, r_endpoints, ntrap=ntrap)
            ic(res_python- res_accelerated)
            assert (abs(res_python-res_accelerated) < 0.001).all()
            return res_accelerated


        if engine == 'accelerated':
            Integral = Integral_Accelerated
        elif engine == 'python':
            Integral = Integral_Python
        elif engine == 'test':
            Integral = Integral_Test
        else:
            assert False # pragma: no cover

        n_endpoints = exnm[1:-1:2,:]
        l_endpoints = n_endpoints[:-1,:]
        r_endpoints = n_endpoints[1:,:]
        vec_delta_l = l_endpoints - r_endpoints

        z = self.jomega * self.mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)

        z *= Integral(l_endpoints, r_endpoints, ntrap=ntrap)

        n_endpoints = exnm[::2,:]
        l_endpoints = n_endpoints[:-1,:]
        r_endpoints = n_endpoints[1:,:]

        s = 1/(self.jomega*self.eps) * Integral(l_endpoints, r_endpoints, ntrap=ntrap)

        z += s[:-1,:-1] + s[1:, 1:] - s[:-1, 1:] - s[1:, :-1]
        
        self.z = z

        return self.factor_and_solve()

    def yagi_compute_impedance(self, *, ntrap=0, engine='python'):

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

        The points themselves are at indices: [2, 4, 6],a
        minus at [1, 3, 5] and plus at [3, 5, 7]
        """

        y, x = np.float64(self.halfdriver), np.float64(self.halfdriver)

        def gen_wire(x, y):
            p0, p1 = np.array((x, -y, 0),dtype=np.float64), np.array((x, y, 0),dtype=np.float64)
            delta_p = (p1-p0)/(2*self.nsegs)
            return np.linspace(p0-delta_p, p1+delta_p, 2*self.nsegs+3)

        wires = (gen_wire(0, y), gen_wire(-x, 1.05*y), gen_wire(x, 0.95*y))

        def Integral_Python(l_endpoints, r_endpoints, ntrap):
            return Integral_Standalone(l_endpoints, r_endpoints, ntrap=ntrap, wire_radius=self.wire_radius, k=self.k)

        Integral = Integral_Python

        l_endpoints = np.empty((0,3))
        r_endpoints = np.empty((0,3))

        for exnm in wires:
            n_endpoints = exnm[1:-1:2,:]
            l_endpoints = np.vstack([l_endpoints, n_endpoints[:-1,:]])
            r_endpoints = np.vstack([r_endpoints, n_endpoints[1:,:]])

        vec_delta_l = l_endpoints - r_endpoints

        z = self.jomega * self.mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)

        z *= Integral(l_endpoints, r_endpoints, ntrap=ntrap)

        l_endpoints = np.empty((0,3))
        r_endpoints = np.empty((0,3))

        for exnm in wires:
            n_endpoints = exnm[::2,:]
            l_endpoints = np.vstack([l_endpoints, n_endpoints[:-1,:]])
            r_endpoints = np.vstack([r_endpoints, n_endpoints[1:,:]])

        s = 1/(self.jomega*self.eps) * Integral(l_endpoints, r_endpoints, ntrap=ntrap)

        d = s[:-1,:-1] + s[1:, 1:] - s[:-1, 1:] - s[1:, :-1]

        off = 0
        for wire in wires[:-1]:
            off += (wire.shape[0]-3)//2
            d = np.delete(d, off, axis=0)
            d = np.delete(d, off, axis=1)

        ic(d.shape)

        z += d
        
        self.z = z

        return self.factor_and_solve()
