import numpy as np
import scipy
import scipy.linalg
from icecream import ic

#from matplotlib import pyplot as plt

ic.disable()

def compute_impedance(wavelength=22, halfdriver_factor=.962):
    wavelength = 22     # meters
    freq = 3e8 / wavelength       # meters/sec / meters = 1/sec = Hz
    omega = freq*2*np.pi          # radians/sec

    k_wavenumber = np.pi*2/wavelength       #radians/meter
    jomega = (0+1j)*omega                   #imaginary radians/sec 

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6
    wire_radius = 0.0005

    halfdriver = halfdriver_factor*wavelength/4

    """
    wire split into to nsegs segments (nsegs + 1 nodes) (2*nsegs + 1 nodes and midpoints)
    """

    nsegs = 101
    y0, y1 = 0, 2*halfdriver

    driver_seg_idx = nsegs//2
    ic(driver_seg_idx)

    """
    Merge segment ends and midpoints into a single array
    """

    nodes_and_midpoints = np.linspace(y0, y1, 2*nsegs+1)
    ic(nodes_and_midpoints)

    def index(pair):
        idx, adj = pair
        res = 2*idx + 1 + adj
        assert 0 <= res < len(nodes_and_midpoints)
        return res

    def diff(a, b):
        return nodes_and_midpoints[index(a)]-nodes_and_midpoints[index(b)]

    def distance(a, b):
        return np.abs(diff(a, b))

    def delta_l(n, *, adj=0):
        if adj == -1:
            return distance((n-1,0), (n, 0))        
        elif adj == 1:
            return distance((n, 0), (n+1, 0))        
        else:
            return distance((n,-1), (n, 1))


    """
    sigma_plus = A_sigma_plus * I
    """


    def Integral(n, m, delta):
        (n_idx, n_adj) = n
        (m_idx, m_adj) = m

        """
    Build coord sys with origin n and the z axis pointing parallel to wire n
    Hack for all wires pointing in y direction
    """
        new_m_coord = (0, 0, diff(m,n))

        if n == m or \
           n_idx+1 == m_idx and n_adj == 1 and m_adj == -1 or \
           m_idx+1 == n_idx and m_adj == 1 and n_adj == -1:
            """
            close integral
            """
            res = 1/(2*np.pi*delta) * np.log(delta/wire_radius) - (0+1j)*k_wavenumber/(4*np.pi)
            #ic('close', n, m, res)
            return res
        else:
            """
            normal integral
            """
            R = np.abs(new_m_coord[2])
            res = np.exp(-(0+1j)*k_wavenumber*R)/(4*np.pi*R)
            #ic('normal', n, m, R, res, np.abs(res), np.angle(res)*180/np.pi)
            return res


    z = np.zeros(shape=(nsegs,nsegs), dtype=np.complex128)

    for m in range(nsegs):
        for n in range(nsegs):
            z[m,n] += jomega * mu * diff((n,-1),(n,1)) * diff((m,-1),(m,1)) * Integral((n, 0), (m, 0), delta_l(n))

            if n+1 < nsegs:
                delta = delta_l(n, adj=1)
            else:
                delta = delta_l(n, adj=0)

            z[m,n] += 1/(jomega*eps) * Integral((n, 1), (m, 1), delta)
            z[m,n] -= 1/(jomega*eps) * Integral((n, 1), (m,-1), delta)

            if 0 < n:
                delta = delta_l(n, adj=-1)
            else:
                delta = delta_l(n, adj=0)

            z[m,n] -= 1/(jomega*eps) * Integral((n,-1), (m, 1), delta)
            z[m,n] += 1/(jomega*eps) * Integral((n,-1), (m,-1), delta)


    ic(z)

    factors = scipy.linalg.lu_factor(z)

    v = np.zeros(shape=(nsegs,), dtype=np.complex128)
    # might need to multiply by segment length, but that does seem to work
    #v[driver_seg_idx] = delta_l(driver_seg_idx) * 1
    v[driver_seg_idx] = 1

    i = scipy.linalg.lu_solve(factors, v)


    ic(factors, v, np.abs(i), np.angle(i)*180/np.pi)
    driver_impedance = 1/i[driver_seg_idx]
    ic(np.abs(driver_impedance), np.angle(driver_impedance)*180/np.pi)

    return driver_impedance, i


# Vector code for distances

def vectorized_compute_impedance(wavelength=22, halfdriver_factor=.962):

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6
    c = 1/np.sqrt(eps*mu)
    ic(eps, mu, c)

    wavelength = 22     # meters
    freq = c / wavelength       # meters/sec / meters = 1/sec = Hz
    omega = freq*2*np.pi          # radians/sec

    k_wavenumber = np.pi*2/wavelength       #radians/meter
    jomega = (0+1j)*omega                   #imaginary radians/sec 

    wire_radius = 0.0005

    halfdriver = halfdriver_factor*wavelength/4

    nsegs = 101
    y0, y1 = 0, 2*halfdriver

    driver_seg_idx = nsegs//2

    p0, p1 = np.array((0, y0, 0)), np.array((0, y1, 0))

    delta_p = (p1-p0)/(2*nsegs)
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
    exnm = np.linspace(p0-delta_p, p1+delta_p, 2*nsegs+3)

    vec_delta_l_minus = exnm[ :-4:2,:] - exnm[2:-2:2,:]
    vec_delta_l       = exnm[1:-3:2,:] - exnm[3:-1:2,:]
    vec_delta_l_plus  = exnm[2:-2:2,:] - exnm[4:  :2,:]

    assert vec_delta_l.shape == (nsegs,3)
    assert vec_delta_l_plus.shape == (nsegs,3)
    assert vec_delta_l_minus.shape == (nsegs,3)

    pts_minus = exnm[1:-3:2,:]
    pts       = exnm[2:-2:2,:]
    pts_plus  = exnm[3:-1:2,:]

    assert pts.shape == (nsegs,3)
    assert pts_plus.shape == (nsegs,3)
    assert pts_minus.shape == (nsegs,3)

    delta_l = np.sqrt((vec_delta_l**2).sum(axis=1))
    delta_l_plus = np.sqrt((vec_delta_l_plus**2).sum(axis=1))
    delta_l_minus = np.sqrt((vec_delta_l_minus**2).sum(axis=1))

    assert delta_l.shape == (nsegs,)
    assert delta_l_plus.shape == (nsegs,)
    assert delta_l_minus.shape == (nsegs,)

    def Integral(n, m, delta):
        R = np.sqrt(((n[np.newaxis, :, :] - m[:, np.newaxis, :])**2).sum(axis=2))

        assert n.shape[0] == delta.shape[0]

        # not always diagonal indices
        diag_indices = np.where(R == 0)
        new_delta = delta[diag_indices[0]]

        RR = R
        RR[diag_indices] = 1
        res = np.exp(-(0+1j)*k_wavenumber*R)/(4*np.pi*RR)
        diag = 1/(2*np.pi*new_delta) * np.log(new_delta/wire_radius) - (0+1j)*k_wavenumber/(4*np.pi) 
        res[diag_indices] = diag
        return res

    z = jomega * mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)

    z *= Integral(pts, pts, delta_l)

    z += 1/(jomega*eps) * Integral(pts_plus, pts_plus, delta_l_plus)
    z -= 1/(jomega*eps) * Integral(pts_plus, pts_minus, delta_l_plus)
    z -= 1/(jomega*eps) * Integral(pts_minus, pts_plus, delta_l_minus)
    z += 1/(jomega*eps) * Integral(pts_minus, pts_minus, delta_l_minus)

    factors = scipy.linalg.lu_factor(z)

    v = np.zeros(shape=(nsegs,), dtype=np.complex128)
    # might need to multiply by segment length, but that does seem to work
    #v[driver_seg_idx] = delta_l(driver_seg_idx) * 1
    v[driver_seg_idx] = 1

    i = scipy.linalg.lu_solve(factors, v)

    ic(np.abs(i), np.angle(i)*180/np.pi)
    driver_impedance = 1/i[driver_seg_idx]
    ic(np.abs(driver_impedance), np.angle(driver_impedance)*180/np.pi)

    return driver_impedance, i
