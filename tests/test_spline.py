
from matplotlib import pyplot as plt
import numpy as np
import scipy
from icecream import ic


fn = None
#fn = '/dev/null'

def gen_matrix(N=3, nrepeats=10, model='fitting', midderivs_free=False):

    constraint = scipy.sparse.dok_array((4*N, 4*N))
    row = 0

    """match f(x) to zero at startpoint"""
    constraint[row, 0:4] = [1, -1/2, 1/4, -1/8]
    row += 1

    """match f(x) between interior points"""
    for i in range(N-1):
        constraint[row, 4*i:4*(i+2)] = [1, 1/2, 1/4, 1/8] + [-1, 1/2, -1/4, 1/8]
        row += 1

    """match f(x) to zero at endpoint"""
    constraint[row, 4*(N-1):4*N] = [1, 1/2, 1/4, 1/8]
    row += 1

    """match f'(x) between splines"""
    for i in range(N-1):
        if not midderivs_free or i+1 != N//2:
            constraint[row, 4*i:4*(i+2)] = [0, 1, 1, 3/4] + [0, -1, 1, -3/4]
            row += 1    

    """match f''(x) between splines"""
    for i in range(N-1):
        if not midderivs_free or i+1 != N//2:
            constraint[row, 4*i:4*(i+2)] = [0, 0, 2, 3] +  [0, 0, -2, 3]
            row += 1

    constraint.resize((row, 4*N))
    constraint = constraint.toarray()
    ic(constraint.shape)

    S = scipy.linalg.null_space(constraint, rcond=1e-8)
    ic(S.shape)
    ic(scipy.linalg.svd(S)[1])

    deriv_op = scipy.sparse.dok_array((4*N, 4*N))
    for i in range(N):
        for j in range(1,4):
            deriv_op[4*i+j-1, 4*i+j] = j

    deriv_op = deriv_op.tocsc()
    deriv2_op = deriv_op @ deriv_op

    #ic(deriv_op.todok())
    #ic(deriv2_op.todok())

    xs = np.linspace(0,N,N*nrepeats+1)
    #ic(xs)

    assert nrepeats % 2 == 0
    coarse_xs = xs[nrepeats//2::nrepeats]

    Vandermonde = scipy.sparse.dok_array((xs.shape[0], 4*N))
    for i,x in enumerate(xs):
        j = min(i // nrepeats, N-1)
        for k in range(4):
            Vandermonde[i,4*j+k] = (x-j-1/2)**k

    Vandermonde = Vandermonde.tocsc()
    ic(Vandermonde.shape)
    ic(scipy.linalg.svd(Vandermonde.toarray())[1])

    if model == 'fit':
        #ys = np.sin(np.pi/N*xs)
        ys = (lambda x: x-x**10)(xs/N)
        ys2 = ys
        eval_mat = Vandermonde @ S
        eval_mat2 = eval_mat
    elif model == 'solve': # solve harrington's DE
        # rhs
        ys = (lambda x: 4*x**2 + 1)(xs/N)
        # solution (should figure out why we need the factor of N*N)
        ys2 = (lambda x: N*N*(5/6*x - x**2/2 - x**4/3))(xs/N)
        eval_mat = Vandermonde @ - deriv2_op @ S
        eval_mat2 = Vandermonde @ S
    elif model == 'vector':
        """Assume I want to create a vector of voltages caused by the current at a bunch of points
    This is a matrix if there is one value representing the current at each point.
    So we can compose the matrices to get a vector of voltages from the vector of spline coefficients.
    V = Z @ (Vandermonde @ S @ coeff)
    S expands the driven coefficients into the full set, and Vandermonde evaluates these coefficient at all the points
"""
        G = scipy.sparse.dok_array((xs.shape[0],xs.shape[0]))
        for i in range(xs.shape[0]):
            #G[i,i] += 1
            if i+1 < xs.shape[0]:
                G[i,i] += 1
                G[i+1,i+1] += 1
                G[i,i+1] -= 1
                G[i+1,i] -= 1

        G = G.tocsc()
        #Z = scipy.sparse.linalg.inv(G)

        ys = np.zeros(xs.shape)
        
        """
        for i in range(ys.shape[0]):
            ys[i] = 1 - 2*abs(i-ys.shape[0]//2)/(ys.shape[0]-1)
"""
        ys[0] = -1/2
        ys[ys.shape[0]//2] = 1
        ys[-1] = -1/2

        ic(G.todok())
        lu = scipy.linalg.lu_factor(G.toarray())
        ys2 = scipy.linalg.lu_solve(lu, ys)

        eval_mat = G @ Vandermonde @ S
        eval_mat2 = Vandermonde @ S

    else:
        assert False # pragma: no cover

    ic(eval_mat.shape)

    def pseudo_solve(A, b):
        U, s, VT = scipy.linalg.svd(A)
        ic(s)

        mask = s > 1e-8
        nnz = np.count_nonzero(mask)
        ic(mask, nnz, VT.shape[0], U.shape[0])
        diag_indices = np.array(range(nnz))

        s_inv = scipy.sparse.coo_array(
            (1/s[:nnz], (diag_indices, diag_indices)),
            shape=(VT.shape[0], U.shape[0])
        )        
        s_inv = s_inv.tocsc()
        ic(s_inv.todok())
        return VT.T @ (s_inv @ (U.T @ b))

    coeffs = pseudo_solve(eval_mat, ys)

    ic(coeffs.shape, coeffs)

    new_ys = eval_mat @ coeffs

    norm = np.sqrt(((new_ys - ys)**2).sum(axis=0))
    ic(norm)

    new_ys2 = eval_mat2 @ coeffs

    return xs, ys, ys2, new_ys, new_ys2

def test_solve():

    N = 4
    nrepeats = 20

    for NN in range(N, N+1, 2):
        xs, ys, ys2, new_ys, new_ys2 = gen_matrix(NN, nrepeats=nrepeats, model='solve')
        plt.plot(xs/N, ys, label='known rhs')
        plt.plot(xs/N, ys2, label='expected solution')
        plt.plot(xs/NN, new_ys, label=f'{NN} predicted rhs')
        plt.plot(xs/NN, new_ys2, label=f'{NN} solution')

        coarse_xs = xs[nrepeats//2::nrepeats]
        coarse_new_ys = new_ys[nrepeats//2::nrepeats]
        plt.plot(coarse_xs/NN, coarse_new_ys, marker='s', linestyle='None')


    plt.legend()
    plt.show()

def test_fit():

    N = 4
    nrepeats = 20
    xs, _, ys, _, _ = gen_matrix(N, nrepeats=nrepeats, model='fit')
    plt.plot(xs/N, ys, label='solution/known')

    for NN in range(N, N+1, 2):
        xs, _, _, new_ys, new_ys2 = gen_matrix(NN, nrepeats=nrepeats, model='fit')
        plt.plot(xs/NN, new_ys, label=f'{NN} predicted')

        coarse_xs = xs[nrepeats//2::nrepeats]
        coarse_new_ys = new_ys[nrepeats//2::nrepeats]
        plt.plot(coarse_xs/NN, coarse_new_ys, marker='s', linestyle='None')


    plt.legend()
    plt.show()

def test_vector():

    N = 4
    nrepeats = 20

    fig, ax0 = plt.subplots()
    ax1 = ax0.twinx()

    for NN in range(N, N+9, 2):
        xs, ys, ys2, new_ys, new_ys2 = gen_matrix(NN, nrepeats=nrepeats, model='vector', midderivs_free=True)

        ax1.plot(xs/NN, ys, label=f'{NN} driven current')
        ax1.plot(xs/NN, ys2, label=f'{NN} expected voltage')
        ax1.plot(xs/NN, new_ys, label=f'{NN} estimated current')
        ax0.plot(xs/NN, new_ys2, label=f'{NN} voltage')

        coarse_xs = xs[nrepeats//2::nrepeats]
        coarse_new_ys = new_ys[nrepeats//2::nrepeats]
        ax1.plot(coarse_xs/NN, coarse_new_ys, marker='s', linestyle='None')


    ax0.legend(loc='upper right')
    ax1.legend(loc='upper left')
    plt.show()
