import numpy as np
import scipy
import scipy.linalg
from icecream import ic

from matplotlib import pyplot as plt

wavelength = 22     # meters
freq = 3e8 / wavelength       # meters/sec / meters = 1/sec = Hz
omega = freq*2*np.pi          # radians/sec

k_wavenumber = np.pi*2/wavelength       #radians/meter
jomega = (0+1j)*omega                   #imaginary radians/sec 

eps = 8.8541878188e-12
mu = 1.25663706127e-6
wire_radius = 0.0005

halfdriver_factor = .962
halfdriver = halfdriver_factor*wavelength/4

"""
wire split into to nsegs segments (nsegs + 1 nodes) (2*nsegs + 1 nodes and midpoints)
"""

nsegs = 101
y0, y1 = 0, 2*halfdriver

nodes = np.linspace(y0, y1, nsegs+1)
ic(nodes)

wires = list(zip(nodes[:-1], nodes[1:]))
ic(wires)

midpoints = np.array([sum(w)/2 for w in wires])
ic(midpoints)

driver_seg_idx = len(wires)//2
driver_seg = wires[driver_seg_idx]
ic(driver_seg_idx, driver_seg)

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

def distance(a, b):
    return abs(nodes_and_midpoints[index(a)]-nodes_and_midpoints[index(b)])

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
    new_m_coord = (0, 0, nodes_and_midpoints[index(m)] - nodes_and_midpoints[index(n)])

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
        z[m,n] += jomega * mu * distance((n,-1),(n,1)) * distance((m,-1),(m,1)) * Integral((n, 0), (m, 0), delta_l(n))

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

#plt.plot(np.real(i))
#plt.show()
