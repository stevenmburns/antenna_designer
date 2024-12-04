import numpy as np
import scipy
import scipy.linalg
from icecream import ic

#from matplotlib import pyplot as plt

class PySim:
    def __init__(self, wavelength=22, halfdriver_factor=.962):
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor

        self.eps = 8.8541878188e-12
        self.mu = 1.25663706127e-6
        self.c = 1/np.sqrt(self.eps*self.mu)

        self.freq = self.c / self.wavelength       # meters/sec / meters = 1/sec = Hz
        self.omega = self.freq*2*np.pi          # radians/sec

        """
        self.k = np.pi*2/self.wavelength       #radians/meter
        self.k = np.pi*2/(self.c/self.freq)
        """
        self.k = self.omega/self.c

        self.jomega = (0+1j)*self.omega                   #imaginary radians/sec 
        self.wire_radius = 0.0005
        self.halfdriver = self.halfdriver_factor*self.wavelength/4

        self.nsegs = 101
        self.driver_seg_idx = self.nsegs//2

    def factor_and_solve(self):
        factors = scipy.linalg.lu_factor(self.z)

        v = np.zeros(shape=(self.nsegs,), dtype=np.complex128)
        v[self.driver_seg_idx] = 1

        i = scipy.linalg.lu_solve(factors, v)

        #ic(factors, v, np.abs(i), np.angle(i)*180/np.pi)
        driver_impedance = 1/i[self.driver_seg_idx]
        ic(np.abs(driver_impedance), np.angle(driver_impedance)*180/np.pi)

        return driver_impedance, i

    def compute_impedance(self):

        """
        wire split into to nsegs segments (nsegs + 1 nodes) (2*nsegs + 1 nodes and midpoints)
        """

        y0, y1 = 0, 2*self.halfdriver

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

        y0, y1 = 0, 2*self.halfdriver

        p0, p1 = np.array((0, y0, 0)), np.array((0, y1, 0))

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
            R = np.sqrt(((n[np.newaxis, :, :] - m[:, np.newaxis, :])**2).sum(axis=2))

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

