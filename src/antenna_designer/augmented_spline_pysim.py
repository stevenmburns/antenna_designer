import numpy as np
import scipy
import scipy.linalg
from icecream import ic

from .abstract_pysim import AbstractPySim

from .spline import NaturalSpline # , PiecewiseLinear
from .pysim_accelerators import psi_fusion_trapezoid

from .pysim import Integral_Standalone

class AugmentedSplinePySim(AbstractPySim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def compute_impedance(self, *, ntrap=0, engine='accelerated', N=4):

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

        factors = scipy.linalg.lu_factor(self.z)

        v = np.zeros(shape=(self.nsegs,), dtype=np.complex128)
        v[self.driver_seg_idx] = 1

        orig_i = scipy.linalg.lu_solve(factors, v)

        spl = NaturalSpline(N=N)
        #spl = PiecewiseLinear(N=N)
        spl.gen_constraint(midderivs_free=True)
        spl.gen_Vandermonde(nsegs=self.nsegs, midpoints=True)

        matched_coeffs = spl.pseudo_solve(spl.Vandermonde @ spl.S, orig_i)

        matched_i = spl.Vandermonde @ spl.S @ matched_coeffs

        ic(np.linalg.norm(orig_i - matched_i))

        matched_v = self.z @ matched_i
        ic(np.linalg.norm(matched_v - v))

        ic(z.shape, spl.Vandermonde.shape, spl.S.shape)

        compressed_z = z @ spl.Vandermonde @ spl.S

        ic(compressed_z.shape, self.nsegs)

        compressed_coeffs = spl.pseudo_solve(compressed_z, v)

        i = spl.Vandermonde @ spl.S @ compressed_coeffs
        ic(i.shape)
        compressed_v = self.z @ i

        ic(np.linalg.norm(matched_i- orig_i), np.linalg.norm(i- orig_i), np.linalg.norm(matched_v - v), np.linalg.norm(compressed_v - v))

        i_driver = i[self.driver_seg_idx]

        driver_impedance = v[self.driver_seg_idx]/i_driver
        ic(np.abs(driver_impedance), np.angle(driver_impedance)*180/np.pi)

        return driver_impedance, (i, orig_i, matched_i)
