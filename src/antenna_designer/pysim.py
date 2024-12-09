import numpy as np
import scipy
import scipy.linalg
from icecream import ic

from numba import jit

from matplotlib import pyplot as plt

class PySim:
    def __init__(self, *, wavelength=22, halfdriver_factor=.962,nsegs=101,rcond=1e-16,nsmallest=0):
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
        run_svd = False
        run_iterative_improvement = False

        factors = scipy.linalg.lu_factor(self.z)

        v = np.zeros(shape=(self.nsegs,), dtype=np.complex128)
        v[self.driver_seg_idx] = 1

        if run_svd:
            i_svd = self.solve_using_svd(self.z, v, rcond=self.rcond, nsmallest=self.nsmallest)

            r =  v - np.dot(self.z, i_svd)
            ic('i_svd error (0)', np.linalg.norm(r))

        i = scipy.linalg.lu_solve(factors, v)

        if run_iterative_improvement:
            i = np.array(i, dtype=np.complex256)
            r =  v - np.dot(self.z, i)
            ic('i error (0)', np.linalg.norm(r))
            i += scipy.linalg.lu_solve(factors,r)

            r =  v - np.dot(self.z, i)
            ic('i error (1)', np.linalg.norm(r))
            i += scipy.linalg.lu_solve(factors,r)

            r =  v - np.dot(self.z, i)
            ic('i error (2)', np.linalg.norm(r))

        if run_svd:
            ic('error vs. svd', np.linalg.norm(i_svd - i))

        #ic(factors, v, np.abs(i), np.angle(i)*180/np.pi)
        driver_impedance = v[self.driver_seg_idx]/i[self.driver_seg_idx]
        ic(np.abs(driver_impedance), np.angle(driver_impedance)*180/np.pi)

        if run_svd:
            driver_impedance_svd = v[self.driver_seg_idx]/i_svd[self.driver_seg_idx]
            ic(np.abs(driver_impedance_svd), np.angle(driver_impedance_svd)*180/np.pi)

        if run_svd:
            return driver_impedance, (i, i_svd)
        else:
            return driver_impedance, i


    def compute_impedance(self):

        """
        wire split into to nsegs segments (nsegs + 1 nodes) (2*nsegs + 1 nodes and midpoints)
        """

        y0, y1 = np.float64(0), np.float64(2*self.halfdriver)

        self.nodes_and_midpoints = np.linspace(y0, y1, 2*self.nsegs+1)

        def index(pair):
            idx, adj = pair
            res = 2*idx + 1 + adj
            assert 0 <= res < len(self.nodes_and_midpoints)
            return res

        def diff(a, b):
            return self.nodes_and_midpoints[index(a)]-self.nodes_and_midpoints[index(b)]

        def distance(a, b):
            return np.abs(diff(a, b))

        def delta_l(n, *, adj=0):
            if adj == -1:
                return distance((n-1,0), (n, 0))        
            elif adj == 1:
                return distance((n, 0), (n+1, 0))        
            else:
                return distance((n,-1), (n, 1))


        def Integral(n, m, delta):
            """
        Build coord sys with origin n and the z axis pointing parallel to wire n
        Hack for all wires pointing in y direction
        """
            new_m_coord = (0, 0, diff(m,n))

            if index(n) == index(m):
                """
                close integral
                """
                res = 1/(2*np.pi*delta) * np.log(delta/self.wire_radius) - (0+1j)*self.k/(4*np.pi)
                return res
            else:
                """
                normal integral
                """
                R = np.abs(new_m_coord[2])
                res = np.exp(-(0+1j)*self.k*R)/(4*np.pi*R)
                return res


        z = np.zeros(shape=(self.nsegs,self.nsegs), dtype=np.complex128)

        for m in range(self.nsegs):
            for n in range(self.nsegs):
                z[m,n] += self.jomega * self.mu * diff((n,-1),(n,1)) * diff((m,-1),(m,1)) * Integral((n, 0), (m, 0), delta_l(n))

                if n+1 < self.nsegs:
                    delta = delta_l(n, adj=1)
                else:
                    delta = delta_l(n, adj=0)

                z[m,n] += 1/(self.jomega*self.eps) * Integral((n, 1), (m, 1), delta)
                z[m,n] -= 1/(self.jomega*self.eps) * Integral((n, 1), (m,-1), delta)

                if 0 < n:
                    delta = delta_l(n, adj=-1)
                else:
                    delta = delta_l(n, adj=0)

                z[m,n] -= 1/(self.jomega*self.eps) * Integral((n,-1), (m, 1), delta)
                z[m,n] += 1/(self.jomega*self.eps) * Integral((n,-1), (m,-1), delta)

        self.z = z
        return self.factor_and_solve()

    def vectorized_compute_impedance(self):

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

        vec_delta_l_minus = exnm[ :-4:2,:] - exnm[2:-2:2,:]
        vec_delta_l       = exnm[1:-3:2,:] - exnm[3:-1:2,:]
        vec_delta_l_plus  = exnm[2:-2:2,:] - exnm[4:  :2,:]

        assert vec_delta_l.shape == (self.nsegs,3)
        assert vec_delta_l_plus.shape == (self.nsegs,3)
        assert vec_delta_l_minus.shape == (self.nsegs,3)

        pts_minus = exnm[1:-3:2,:]
        pts       = exnm[2:-2:2,:]
        pts_plus  = exnm[3:-1:2,:]

        assert pts.shape == (self.nsegs,3)
        assert pts_plus.shape == (self.nsegs,3)
        assert pts_minus.shape == (self.nsegs,3)

        delta_l = np.sqrt((vec_delta_l**2).sum(axis=1))
        delta_l_plus = np.sqrt((vec_delta_l_plus**2).sum(axis=1))
        delta_l_minus = np.sqrt((vec_delta_l_minus**2).sum(axis=1))

        assert delta_l.shape == (self.nsegs,)
        assert delta_l_plus.shape == (self.nsegs,)
        assert delta_l_minus.shape == (self.nsegs,)

        def Integral(n, m, delta):

            diffs = n[np.newaxis, :, :] - m[:, np.newaxis, :]
            R = np.sqrt((diffs*diffs).sum(axis=2))

            assert n.shape[0] == delta.shape[0]

            # not always diagonal indices
            diag_indices = np.where(R == 0)
            new_delta = delta[diag_indices[0]]

            RR = R
            RR[diag_indices] = 1
 
            res = np.exp(-(0+1j)*self.k*R)/(4*np.pi*RR)
            diag = 1/(2*np.pi*new_delta) * np.log(new_delta/self.wire_radius) - (0+1j)*self.k/(4*np.pi) 
            res[diag_indices] = diag

            return res

        z = self.jomega * self.mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)

        z *= Integral(pts, pts, delta_l)

        z += 1/(self.jomega*self.eps) * Integral(pts_plus, pts_plus, delta_l_plus)
        z -= 1/(self.jomega*self.eps) * Integral(pts_plus, pts_minus, delta_l_plus)
        z -= 1/(self.jomega*self.eps) * Integral(pts_minus, pts_plus, delta_l_minus)
        z += 1/(self.jomega*self.eps) * Integral(pts_minus, pts_minus, delta_l_minus)

        self.z = z

        return self.factor_and_solve()

    def stamp_vectorized_compute_impedance(self):

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

        pts = np.linspace(p0+delta_p, p1-delta_p, self.nsegs)

        extras = np.linspace(p0, p1, self.nsegs+1)

        vec_delta_l = extras[1:] - extras[:-1]
        assert vec_delta_l.shape == (self.nsegs,3)
        delta_l = np.sqrt((vec_delta_l**2).sum(axis=1))
        assert delta_l.shape == (self.nsegs,)

        # hack that works for equal size segments
        delta_l_extras = delta_l[0] * np.ones(shape=(extras.shape[0],))
        assert delta_l_extras.shape == (self.nsegs+1,)

        def Integral(n, m, delta):

            diffs = n[np.newaxis, :, :] - m[:, np.newaxis, :]
            R = np.sqrt((diffs*diffs).sum(axis=2))

            assert n.shape[0] == delta.shape[0]

            # not always diagonal indices
            diag_indices = np.where(R == 0)
            new_delta = delta[diag_indices[0]]

            RR = R
            RR[diag_indices] = 1
 
            res = np.exp(-(0+1j)*self.k*R)/(4*np.pi*RR)
            diag = 1/(2*np.pi*new_delta) * np.log(new_delta/self.wire_radius) - (0+1j)*self.k/(4*np.pi) 
            res[diag_indices] = diag

            return res

        z = self.jomega * self.mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)

        z *= Integral(pts, pts, delta_l)

        s = 1/(self.jomega*self.eps) * Integral(extras, extras, delta_l_extras)

        z += s[:-1,:-1] + s[1:, 1:] - s[:-1, 1:] - s[1:, :-1]

        self.z = z

        return self.factor_and_solve()

    def interpolated_compute_impedance(self):

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

        vec_delta_l_minus = exnm[ :-4:2,:] - exnm[2:-2:2,:]
        vec_delta_l       = exnm[1:-3:2,:] - exnm[3:-1:2,:]
        vec_delta_l_plus  = exnm[2:-2:2,:] - exnm[4:  :2,:]

        assert vec_delta_l.shape == (self.nsegs,3)
        assert vec_delta_l_plus.shape == (self.nsegs,3)
        assert vec_delta_l_minus.shape == (self.nsegs,3)

        pts_minus = exnm[1:-3:2,:]
        pts       = exnm[2:-2:2,:]
        pts_plus  = exnm[3:-1:2,:]

        assert pts.shape == (self.nsegs,3)
        assert pts_plus.shape == (self.nsegs,3)
        assert pts_minus.shape == (self.nsegs,3)

        delta_l = np.sqrt((vec_delta_l**2).sum(axis=1))
        delta_l_plus = np.sqrt((vec_delta_l_plus**2).sum(axis=1))
        delta_l_minus = np.sqrt((vec_delta_l_minus**2).sum(axis=1))

        assert delta_l.shape == (self.nsegs,)
        assert delta_l_plus.shape == (self.nsegs,)
        assert delta_l_minus.shape == (self.nsegs,)

        def IntegralSlow(n, m, delta):

            diffs = n[np.newaxis, :, :] - m[:, np.newaxis, :]
            R = np.sqrt((diffs*diffs).sum(axis=2))

            assert n.shape[0] == delta.shape[0]

            # not always diagonal indices
            diag_indices = np.where(R == 0)
            new_delta = delta[diag_indices[0]]

            RR = R
            RR[diag_indices] = 1
            res = np.exp(-(0+1j)*self.k*R)/(4*np.pi*RR)
            diag = 1/(2*np.pi*new_delta) * np.log(new_delta/self.wire_radius) - (0+1j)*self.k/(4*np.pi) 
            res[diag_indices] = diag

            return res

        def IntegralPoint(n, m, delta):
            # we want the integral with y ranging from a to b
            # currently being done with (b-a)*f((a+b)/2) [midpoint rule/rectangle rule]
            # The trapezodial rule is (b-a)*(f(a)+f(b))/2
            # The composite trapezoidal rule is (b-a)/N*f(f(a)/2 + sum_k=1^N-1 f(a+k(b-a)/N) + f(b)/2)

            # looks like we need a delta_m, because delta belongs to n.
            # probably it would be better to use indices into exnm, then we can get the delta from that array

            # if we know delta_m- and delta_m+, then we can compute a and b
            # a = m - delta_m-/2 and b = m + delta_m+/2
            #
            # Then b-a = delta_m+/2 + delta_m-/2
            # N=0 (b-a)*(f((a+b)/2)
            # N=1 (b-a)*(f(a)+f(b))/2
            # N=2 (b-a)*(f(a)+2*f((a+b)/2)+f(b))/4
            # N=3 (b-a)*(f(a)+2*f(2*a/3+b/3)+2*f(a/3+2*b/3)+f(b))/6
            # N=4 (b-a)*(f(a)+2*f(3*a/4+b/4)+2*f(a/2+b/2)+2*f(a/4+3*b/4)+f(b))/8
            #
            # for N=2 we can pick things out of exnum
            # exnum[m_idx] is (a+b)/2
            # exnum[m_idx-1] is a
            # exnum[m_idx+1] is b
            # vec_delta_m- = exnm[m_idx-2] - exnm[m_idx]
            # vec_delta_m+ = exnm[m_idx] - exnm[m_idx+2]

            # compute R for exnm[n_idx] to exnm[m_idx-2], exnm[m_idx], and exnm[m_idx+2]


            diffs = n[np.newaxis, :, :] - m[:, np.newaxis, :]
            R = np.sqrt((diffs*diffs).sum(axis=2))

            assert n.shape[0] == delta.shape[0]

            # not always diagonal indices
            diag_indices = np.where(R == 0)
            new_delta = delta[diag_indices[0]]

            RR = R
            RR[diag_indices] = 1
            res = delta[:, np.newaxis] * np.exp(-(0+1j)*self.k*R)/(4*np.pi*RR)
            diag = 1/(2*np.pi*new_delta) * np.log(new_delta/self.wire_radius) - (0+1j)*self.k/(4*np.pi) 
            res[diag_indices] = diag

            return res

        diffs = pts_minus[np.newaxis, :, :] - pts_plus[:, np.newaxis, :]
        R = np.sqrt((diffs*diffs).sum(axis=2))
        max_R = np.max(R)

        rs = np.linspace(0, max_R, 1001)

        tbl = IntegralSlow(np.array([[0,0,0]]), np.array([[0, r, 0] for r in rs]), delta_l[:1])[:,0]
        ic(tbl.shape)

        if False:
            plt.plot(rs,np.real(tbl))
            plt.plot(rs,np.imag(tbl))

            rs = np.linspace(0, max_R, 101)

            tbl = IntegralSlow(np.array([[0,0,0]]), np.array([[0, r, 0] for r in rs]), delta_l[:1])[:,0]
            ic(tbl.shape)

            ic(R.shape, rs.shape, tbl.shape)
            plt.plot(rs,np.real(tbl))
            plt.plot(rs,np.imag(tbl))
            plt.show()

        def Integral(n, m, delta):
            diffs = n[np.newaxis, :, :] - m[:, np.newaxis, :]
            R = np.sqrt((diffs*diffs).sum(axis=2))
            return np.interp(R, rs, tbl)

        z = self.jomega * self.mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)

        z *= Integral(pts, pts, delta_l)

        z += 1/(self.jomega*self.eps) * Integral(pts_plus, pts_plus, delta_l_plus)
        z -= 1/(self.jomega*self.eps) * Integral(pts_plus, pts_minus, delta_l_plus)
        z -= 1/(self.jomega*self.eps) * Integral(pts_minus, pts_plus, delta_l_minus)
        z += 1/(self.jomega*self.eps) * Integral(pts_minus, pts_minus, delta_l_minus)

        self.z = z

        return self.factor_and_solve()
