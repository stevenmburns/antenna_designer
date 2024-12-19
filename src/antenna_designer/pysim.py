import numpy as np
import scipy
import scipy.linalg
from icecream import ic

from .pysim_accelerators import psi_fusion_trapezoid

from .spline import NaturalSpline, PiecewiseLinear

class PySim:
    def __init__(self, *, wavelength=22, halfdriver_factor=.962,nsegs=101,rcond=1e-16,nsmallest=0, run_iterative_improvement=False, run_svd=False):
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.nsegs = nsegs
        self.rcond = rcond
        self.nsmallest = nsmallest

        self.eps = 8.8541878188e-12
        self.mu = 1.25663706127e-6
        self.c = 1/np.sqrt(self.eps*self.mu)

        self.freq = self.c / self.wavelength  # meters/sec / meters = 1/sec = Hz
        self.omega = self.freq*2*np.pi        # radians/sec

        """
        self.k = np.pi*2/self.wavelength      #radians/meter
        self.k = np.pi*2/(self.c/self.freq)
        """
        self.k = self.omega/self.c

        self.jomega = (0+1j)*self.omega       #imaginary radians/sec 
        self.wire_radius = 0.0005
        self.halfdriver = self.halfdriver_factor*self.wavelength/4


        self.driver_seg_idx = self.nsegs//2

        self.run_svd = run_svd
        self.run_iterative_improvement = run_iterative_improvement


    @staticmethod
    def solve_using_svd(A, b, rcond=1e-16, nsmallest=0):
        u, s, vh = scipy.linalg.svd(A)
        abss = np.abs(s)

        if nsmallest > 0:
            # sorted in decreasing order?
            assert np.all(abss[1:] <= abss[:-1])
            ic(abss, abss[-nsmallest])
            mask = abss <= abss[-nsmallest]
        else:
            mask = abss > rcond * np.max(abss)

        ic(np.max(abss)/np.min(abss), np.count_nonzero(mask))

        u, s, vh = u[:,mask], s[mask], vh[mask,:]

        def solve(b):
            return vh.conj().T @ (np.diag(1/s) @ (u.T @ b))

        x = solve(b)

        if False:
            x = np.array(x, dtype=np.complex256)

            r = b - A@x
            ic('svd residual norm (0)', np.linalg.norm(r))
            x += solve(r)
            r = b - A@x
            ic('svd residual norm (1)', np.linalg.norm(r))
            x += solve(r)
            r = b - A@x
            ic('svd residual norm (2)', np.linalg.norm(r))

        return x

    def factor_and_solve(self):
        factors = scipy.linalg.lu_factor(self.z)

        v = np.zeros(shape=(self.nsegs,), dtype=np.complex128)
        v[self.driver_seg_idx] = 1

        if self.run_svd:
            i_svd = self.solve_using_svd(self.z, v, rcond=self.rcond, nsmallest=self.nsmallest)

            r =  v - np.dot(self.z, i_svd)
            ic('i_svd error (0)', np.linalg.norm(r))

        i = scipy.linalg.lu_solve(factors, v)

        if self.run_iterative_improvement:
            i = np.array(i, dtype=np.complex256)
            r =  v - np.dot(self.z, i)
            ic('i error (0)', np.linalg.norm(r))
            i += scipy.linalg.lu_solve(factors,r)

            r =  v - np.dot(self.z, i)
            ic('i error (1)', np.linalg.norm(r))
            i += scipy.linalg.lu_solve(factors,r)

            r =  v - np.dot(self.z, i)
            ic('i error (2)', np.linalg.norm(r))

        if self.run_svd:
            ic('error vs. svd', np.linalg.norm(i_svd - i))

        #ic(factors, v, np.abs(i), np.angle(i)*180/np.pi)
        driver_impedance = v[self.driver_seg_idx]/i[self.driver_seg_idx]
        ic(np.abs(driver_impedance), np.angle(driver_impedance)*180/np.pi)

        if self.run_svd:
            driver_impedance_svd = v[self.driver_seg_idx]/i_svd[self.driver_seg_idx]
            ic(np.abs(driver_impedance_svd), np.angle(driver_impedance_svd)*180/np.pi)

        if self.run_svd:
            return driver_impedance, (i, i_svd)
        else:
            return driver_impedance, i


    def augmented_compute_impedance(self, *, ntrap=0, engine='accelerated'):

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


    def augmented_spline_compute_impedance(self, *, ntrap=0, engine='accelerated', N=4):

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

        matched_i = spl.Vandermonde @ spl.S @ spl.pseudo_solve(spl.Vandermonde @ spl.S, orig_i)

        ic(np.linalg.norm(orig_i - matched_i))

        ic(z.shape, spl.Vandermonde.shape, spl.S.shape)

        compressed_z = z @ spl.Vandermonde @ spl.S

        ic(compressed_z.shape, self.nsegs)


        i = spl.Vandermonde @ spl.S @ spl.pseudo_solve(compressed_z, v)
        ic(i.shape)

        i_driver = i[self.driver_seg_idx]

        driver_impedance = v[self.driver_seg_idx]/i_driver
        ic(np.abs(driver_impedance), np.angle(driver_impedance)*180/np.pi)

        return driver_impedance, (i, orig_i, matched_i)


    def spline_compute_impedance(self, *, ntrap=0, engine='python', N=20):

        y0, y1 = np.float64(0), np.float64(2*self.halfdriver)

        p0, p1 = np.array((0, y0, 0),dtype=np.float64), np.array((0, y1, 0),dtype=np.float64)

        spl = NaturalSpline(N=N)
        #spl = PiecewiseLinear(N=N)
        spl.gen_constraint(midderivs_free=True)
        spl.gen_Vandermonde(nsegs=self.nsegs*ntrap)
        spl.gen_deriv_ops()

        n_start = p0
        n_end   = p1
        total_length = np.sqrt((n_end-n_start)**2).sum(axis=0)
        ic(total_length)

        xs = np.linspace(n_start, n_end, self.nsegs*ntrap+1)
        #ic(xs)

        m_endpoints = np.linspace(n_start, n_end, self.nsegs)
        m_centers = 0.5*(m_endpoints[1:, :] + m_endpoints[:-1, :])
        m_delta = m_endpoints[1:, :] - m_endpoints[:-1, :]

        # m are the test points (can be -1, 0, or +1) because we need to take numerical derivatives of the voltage
        # n are the integration intervals (centers on 1/2 points) tied to the spline

        def Spline_Integral_Standalone(m, *, ntrap, wire_radius, k, use_deriv):
            assert ntrap != 0

            m_vec_delta = m[1:, :] - m[:-1, :]
            delta = np.sqrt((m_vec_delta*m_vec_delta).sum(axis=1))
            assert m.shape[0] - 1 == delta.shape[0]


            if use_deriv:
                eval_mat = spl.Vandermonde @ (N/self.nsegs*spl.deriv_op) @ spl.S
            else:
                eval_mat = spl.Vandermonde @ spl.S
            ic(eval_mat.shape)

            def Aux(i):
                local_n = xs[i:len(xs)-ntrap+i:ntrap,:]

                ic(i, local_n.shape, m.shape)

                diffs = local_n[np.newaxis, :, :] - m[:, np.newaxis, :]
                R = np.sqrt((diffs*diffs).sum(axis=2))

                ic(R.shape)

                # not always diagonal indices
                diag_indices = np.where(R < 0.00001)
                ic(diag_indices[0].shape, diag_indices[1].shape, diag_indices[0], delta.shape)

                #new_delta = delta[diag_indices[0]]
                #hack assuming all deltas are identical
                new_delta = delta[0]

                RR = R
                RR[diag_indices] = 1

                local_res = np.exp(-(0+1j)*k*R)/(4*np.pi*RR)
                diag = 1/(2*np.pi*new_delta) * np.log(new_delta/wire_radius) - (0+1j)*k/(4*np.pi) 
                local_res[diag_indices] = diag

                ic(local_res.shape)

                restricted_eval_mat = eval_mat[i:len(xs)-ntrap+i:ntrap,:] 
                ic(restricted_eval_mat.shape)

                tmp = local_res @ restricted_eval_mat
                ic(tmp.shape)
                return tmp

            res = np.zeros(shape=(m.shape[0], spl.S.shape[1]),dtype=np.complex128)
            ic(res.shape)
            for i in range(0, ntrap+1):
                coeff = (2 if i > 0 and i < ntrap else 1)/(2*ntrap)
                res += coeff * Aux(i)


            return res


        def Integral(m, ntrap, use_deriv):
            return Spline_Integral_Standalone(m, ntrap=ntrap, wire_radius=self.wire_radius, k=self.k, use_deriv=use_deriv)

        #z = self.jomega * self.mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)
        z = self.jomega * self.mu * (m_delta[0, :] * m_delta[0, :]).sum(axis=0)

        z *= Integral(m_centers, ntrap=ntrap, use_deriv=False)

        s = 1/(self.jomega*self.eps) * Integral(m_endpoints, ntrap=ntrap, use_deriv=True)

        z += s[1:, :] - s[:-1, :]

        self.z = z

        ic(z.shape, self.nsegs)

        v = np.zeros(shape=(self.z.shape[0],), dtype=np.complex128)
        v[self.driver_seg_idx] = 1

        i = spl.Vandermonde @ spl.S @ spl.pseudo_solve(self.z, v)

        i_driver = i[i.shape[0]//2]
        #i_driver = i[self.driver_seg_idx]
        ic(i.shape)

        driver_impedance = v[self.driver_seg_idx]/i_driver
        ic(np.abs(driver_impedance), np.angle(driver_impedance)*180/np.pi)

        return driver_impedance, i

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


