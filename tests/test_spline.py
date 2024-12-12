
from matplotlib import pyplot as plt
import numpy as np
import scipy
from icecream import ic


fn = None
#fn = '/dev/null'

def gen_matrix(N=3):

    constraint = scipy.sparse.dok_array((4*N,4*N))
    row = 0

    """match f(x) at startpoint"""
    constraint[row, 0:4] = [1, -1/2, 1/4, -1/8]
    row += 1

    """match f(x) between interior points"""
    for i in range(N-1):
        constraint[row, 4*i:4*(i+2)] = [1, 1/2, 1/4, 1/8] + [-1, 1/2, -1/4, 1/8]
        row += 1

    """match f(x) at endpoint"""
    constraint[row, 4*(N-1):4*N] = [1, 1/2, 1/4, 1/8]
    row += 1

    """match f'(x) between splines"""
    for i in range(N-1):
        constraint[row, 4*i:4*(i+2)] = [0, 1, 1, 3/4] + [0, -1, 1, -3/4]
        row += 1    

    """match f''(x) between splines"""
    for i in range(N-1):
        constraint[row, 4*i:4*(i+2)] = [0, 0, 2, 3] +  [0, 0, -2, 3]
        row += 1    

    """make these driven"""
    for i in range(N):    
        constraint[row, 4*i:4*(i+1)] = [1, 0, 0, 0]
        row += 1    

    """make f''(x) on left boundary driven"""
    constraint[row, 0:4] = [0, 0, 2, 3]
    row += 1    

    ic(constraint)

    constrain_coeffs = constraint.toarray()

    inv = np.linalg.inv(constrain_coeffs)

    # one driving variable for each N and the remaining second derivative
    S = inv[:, -(N+1):]
    ic(S)

    nrepeats = 10
    xs = np.linspace(0,N,N*nrepeats+1)
    ic(xs)

    v = scipy.sparse.dok_array((xs.shape[0], 4*N))
    for i,x in enumerate(xs):
        j = min(i // nrepeats, N-1)
        for k in range(4):
            v[i,4*j+k] = (x-j-1/2)**k

    v = v.tocsr()

    ic(v)

    #ys = np.sin(np.pi/N*xs)
    ys = (lambda x: x-x**10)(xs/N)

    eval_mat = v @ S

    ic(eval_mat)

    def pseudo_solve(A, b):
        U, s, VT = scipy.linalg.svd(A)
        ic(s)
        diag_indices = np.array(range(s.shape[0]))
        s_inv = scipy.sparse.coo_array(
            (1/s, (diag_indices, diag_indices)),
            shape=(VT.shape[0], U.shape[0])
        )        
        s_inv = s_inv.tocsr()
        return VT.T @ (s_inv @ (U.T @ b))

    coeffs = pseudo_solve(eval_mat, ys)

    ic(coeffs.shape, coeffs)

    new_ys = eval_mat @ coeffs

    return xs, ys, new_ys, coeffs

def test_gen_matrix():

    N = 3
    xs, ys, _, _ = gen_matrix(N)
    plt.plot(xs/N, ys)

    for N in range(3, 9):
        xs, _, new_ys, coeffs = gen_matrix(N)
        plt.plot(xs/N, new_ys)

        coeffs = coeffs[:-1]
        delta = 1/(N)
        ic(N, coeffs.shape, delta)

        xxs = np.linspace(delta/2, 1-delta/2, N)
        ic(xxs)
        plt.plot(xxs, coeffs, marker='s')



    plt.show()
