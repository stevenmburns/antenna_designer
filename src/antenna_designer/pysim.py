import numpy as np
from icecream import ic

"""
wire split into to nsegs segments (nsegs + 1 nodes) (2*nsegs + 1 nodes and midpoints)
"""

nsegs = 5

y0, y1 = 0, 5

nodes = np.linspace(y0, y1, nsegs+1)
ic(nodes)

wires = list(zip(nodes[:-1], nodes[1:]))
ic(wires)

midpoints = np.array([sum(w)/2 for w in wires])
ic(midpoints)

driver_wire = wires[len(wires)//2]
ic(driver_wire)

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


wavelength = 11*4
freq = 3e8 / wavelength
omega = freq*2*np.pi

omega = 1
freq = omega/(2*np.pi)
wavelength = 3e8 / freq

k_wavenumber = np.pi*2/wavelength
jomega = (0+1j)*omega


"""
1 = freq*2*pi    [radians per sec]
freq = 1/(2*pi)  [cycles per sec]
wavelength/2*pi = c       [meters per sec]
wavelength = c*2*pi       [meters]
k = 2*pi/(c*2*pi) = 1 / c  [sec per meter]
"""



A_sigma_plus = np.zeros(shape=(nsegs,nsegs), dtype=np.complex128)
for n in range(nsegs):
    if n+1 < nsegs:
        A_sigma_plus[n,n+1] = -1 / jomega / delta_l(n, adj=1)
    A_sigma_plus[n,n] = 1 / jomega / delta_l(n, adj=0)

A_sigma_minus = np.zeros(shape=(nsegs,nsegs), dtype=np.complex128)
for n in range(nsegs):
    A_sigma_minus[n,n] = -1 / jomega / delta_l(n, adj=0)
    if 0 < n:
        A_sigma_minus[n,n-1] = 1 / jomega / delta_l(n, adj=-1)

"""
Boundaries might not be right.
For the bottom and top diagonal entries, respectively, I used the closest defined value of delta_l.
"""

ic(A_sigma_plus)
ic(A_sigma_minus)

eps = 1
mu = 1

"""
Convert to phi_plus from sigma_plus
"""

def Integral(n, m, delta):
    (n_idx, n_adj) = n
    (m_idx, m_adj) = m


    """
Build coord sys with origin n and the z axis pointing parallel to wire n
"""
    new_m_coord = (0, 0, nodes_and_midpoints[index(m)] - nodes_and_midpoints[index(n)])

    if n == m or \
       n_idx+1 == m_idx and n_adj == 1 and m_adj == -1 or \
       m_idx+1 == n_idx and m_adj == 1 and n_adj == -1:
        """
        close integral
        """
        ic('close', n, m)
        pass
    else:
        """
        normal integral
        """
        ic('normal', n, m)
        pass

    return 1/eps


B_phi_plus = np.zeros(shape=(nsegs,nsegs), dtype=np.complex128)
for m in range(nsegs):
    for n in range(nsegs):
        if n+1 < nsegs:
            B_phi_plus[m,n] = Integral( (m,1), (n,1), delta_l(n, adj=1))
        else:
            B_phi_plus[m,n] = Integral( (m,1), (n,1), delta_l(n, adj=0))

B_phi_minus = np.zeros(shape=(nsegs,nsegs), dtype=np.complex128)
for m in range(nsegs):
    for n in range(nsegs):
        if 0 < n:
            B_phi_minus[m,n] = Integral( (m,-1), (n,-1), delta_l(n, adj=-1))
        else:
            B_phi_minus[m,n] = Integral( (m,-1), (n,-1), delta_l(n, adj=0))

ic(B_phi_plus)
ic(B_phi_minus)
