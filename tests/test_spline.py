
from matplotlib import pyplot as plt
import numpy as np
import scipy
from icecream import ic


fn = None
#fn = '/dev/null'

def gen_matrix(N=3):

    rows = []

    """make f(x) 0 on left boundary"""
    rows.append( [1, -1/2, 1/4, -1/8] + [0]*4*(N-1) )

    """match f(x) between splines"""
    for i in range(N-1):
        rows.append( [0]*4*i + [1, 1/2, 1/4, 1/8] +  [-1, 1/2, -1/4, 1/8] + [0]*4*(N-2-i) )

    """make f(x) 0 on right boundary"""
    rows.append( [0]*4*(N-1) + [1, 1/2, 1/4, 1/8] )

    """match f'(x) between splines"""
    for i in range(N-1):
        rows.append( [0]*4*i + [0, 1, 1, 3/4] +  [0, -1, 1, -3/4] + [0]*4*(N-2-i) )

    """match f''(x) between splines"""
    for i in range(N-1):
        rows.append( [0]*4*i + [0, 0, 2, 3] +  [0, 0, -2, 3] + [0]*4*(N-2-i) )


    """make these driven"""
    for i in range(N):    
        rows.append( [0]*4*i + [1, 0, 0, 0] + [0]*4*(N-1-i) )

    """make f''(x) on left boundary driven"""
    rows.append( [0, 0, 2, 3] + [0]*4*(N-1) )
        


    ic(len(rows), len(rows[0]), rows)

    constrain_coeffs = np.array(rows)

    inv = np.linalg.inv(constrain_coeffs)

    # one driving variable for each N and the remaining second derivative
    S = inv[:, -(N+1):]
    ic(S)

    nrepeats = 10
    xs = np.linspace(0,N,N*nrepeats+1)
    ic(xs)

    v = np.zeros(shape=(xs.shape[0],4*N))
    for i,x in enumerate(xs):
        j = min(i // nrepeats, N-1)
        for k in range(4):
            v[i,4*j+k] = (x-j-1/2)**k
    
    ic(v)

    #ys = np.sin(np.pi/N*xs)
    ys = (lambda x: x-x**10)(xs/N)

    eval_mat = v @ S

    def pseudo_inverse(A):
        U, s, VT = scipy.linalg.svd(A)
        ic(s)
        s_inv = np.zeros(shape=(VT.shape[0], U.shape[0]))
        s_inv[:s.shape[0], :s.shape[0]] = np.diag(1/s)
        return VT.T @ s_inv @ U.T

    coeffs = pseudo_inverse(eval_mat) @ ys

    ic(coeffs.shape, coeffs)

    new_ys = eval_mat @ coeffs

    return xs, ys, new_ys

def test_gen_matrix():

    N = 3
    xs, ys, new_ys = gen_matrix(N)
    plt.plot(xs/N, ys)

    for N in range(4, 10):
        xs, _, new_ys = gen_matrix(N)
        plt.plot(xs/N, new_ys)



    plt.show()
