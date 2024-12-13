
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

    """make another variable driven (left most a1 term)"""
    constraint[row, 0:4] = [0, 1, 0, 0]
    row += 1    
    
    
    """Reorder to put driven variables at the end"""
    driven = [4*i for i in range(N)] + [1]

    set_driven = set(driven)
    lst = []
    for i in range(4*N):
        if i not in set_driven:
            lst.append(i)

    order = lst + driven

    ic(order)

    constraint = constraint.tocsc()[:, order]

    ic(constraint)

    lu = scipy.sparse.linalg.splu(constraint[:3*N-1, :3*N-1])
    ic(lu.shape, lu.nnz, lu)

    SS = -lu.solve(constraint[:3*N-1, 3*N-1:].toarray())
    SSS = np.vstack((SS, np.eye(N+1)))

    inv = np.linalg.inv(constraint.toarray())
    # one driving variable for each N and the remaining second derivative
    S = inv[:, -(N+1):]

    assert (np.abs(SSS - S) < 0.0001).all()

    ic(inv, S, S.shape, SSS, SSS.shape)

    deriv_op = scipy.sparse.dok_array((4*N, 4*N))
    for i in range(N):
        for j in range(1,4):
            deriv_op[4*i+j-1, 4*i+j] = j

    deriv2_op = deriv_op @ deriv_op

    deriv_op = deriv_op.tocsc()[:, order][order, :]
    deriv2_op = deriv2_op.tocsc()[:, order][order, :]

    ic(deriv_op.todok())
    ic(deriv2_op.todok())

    nrepeats = 10
    xs = np.linspace(0,N,N*nrepeats+1)
    ic(xs)

    Vandermonde = scipy.sparse.dok_array((xs.shape[0], 4*N))
    for i,x in enumerate(xs):
        j = min(i // nrepeats, N-1)
        for k in range(4):
            Vandermonde[i,4*j+k] = (x-j-1/2)**k

    ic(Vandermonde)
    Vandermonde = Vandermonde.tocsc()[:,order]

    if False: # fitting
        #ys = np.sin(np.pi/N*xs)
        ys = (lambda x: x-x**10)(xs/N)
        ys2 = ys
        eval_mat = Vandermonde @ S
        eval_mat2 = eval_mat
    else: # solve harrington's DE
        # rhs
        ys = (lambda x: 4*x**2 + 1)(xs/N)
        # solution (should figure out why we need the factor of N*N)
        ys2 = (lambda x: N*N*(5/6*x - x**2/2 - x**4/3))(xs/N)
        eval_mat = Vandermonde @ - deriv2_op @ S
        eval_mat2 = Vandermonde @ S

    ic(eval_mat)

    def pseudo_solve(A, b):
        U, s, VT = scipy.linalg.svd(A)
        ic(s)
        diag_indices = np.array(range(s.shape[0]))
        s_inv = scipy.sparse.coo_array(
            (1/s, (diag_indices, diag_indices)),
            shape=(VT.shape[0], U.shape[0])
        )        
        s_inv = s_inv.tocsc()
        return VT.T @ (s_inv @ (U.T @ b))

    coeffs = pseudo_solve(eval_mat, ys)

    ic(coeffs.shape, coeffs)

    new_ys = eval_mat2 @ coeffs

    return xs, ys2, new_ys, coeffs

def test_gen_matrix():

    N = 3
    xs, ys, _, _ = gen_matrix(N)
    plt.plot(xs/N, ys, label='solution/known')

    for N in range(3, 4):
        xs, _, new_ys, coeffs = gen_matrix(N)
        plt.plot(xs/N, new_ys, label=f'{N} predicted')

        coeffs = coeffs[:-1]
        delta = 1/(N)
        ic(N, coeffs.shape, delta)

        xxs = np.linspace(delta/2, 1-delta/2, N)
        ic(xxs)
        plt.plot(xxs, coeffs, marker='s', linestyle='None')


    plt.legend()
    plt.show()
