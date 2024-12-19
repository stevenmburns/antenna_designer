
import numpy as np
import scipy
from icecream import ic


class AbstractSpline:
    def __init__(self, N=3, degree=3):
        self.N = N
        self.degree = degree

    def gen_deriv_ops(self):
        N = self.N
        D = self.degree

        self.deriv_op = scipy.sparse.dok_array((D*N, D*N))
        for i in range(N):
            for j in range(1,D):
                self.deriv_op[D*i+j-1, D*i+j] = j

        self.deriv_op = self.deriv_op.tocsc()
        self.deriv2_op = self.deriv_op @ self.deriv_op

        #ic(self.deriv_op.todok())
        #ic(self.deriv2_op.todok())

    def gen_Vandermonde(self, nsegs, midpoints=False):
        self.nsegs = nsegs
        N = self.N
        D = self.degree

        if midpoints:
            delta = N/nsegs
            self.xs = np.linspace(delta/2,N-delta/2,nsegs)            
        else:
            self.xs = np.linspace(0,N,nsegs+1)
        #ic(self.xs)

        # There are N splines and nsegs segments
        # what are the coordinates of the segment endpoints?

        # if N is 6 and nsegs is 9,
        spline_idx = (self.xs + 0.0001).astype(np.int64)
        ic(spline_idx, spline_idx.dtype)
        spline_idx[-1] = N-1
        remainder = self.xs - spline_idx
        ic(spline_idx, remainder, self.xs)
        
        self.Vandermonde = scipy.sparse.dok_array((self.xs.shape[0], D*N))
        for i,(x,j) in enumerate(zip(self.xs,spline_idx)):
            for k in range(D):
                self.Vandermonde[i,D*j+k] = (x-j-1/2)**k

        self.Vandermonde = self.Vandermonde.tocsc()
        ic(self.Vandermonde.shape)
        ic(scipy.linalg.svd(self.Vandermonde.toarray())[1])

    @staticmethod
    def pseudo_solve(A, b):
        ic(A.shape, b.shape)
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

    def backend(self, *, eval_mat, rhs):
        ic(eval_mat.shape)

        coeffs = self.pseudo_solve(eval_mat, rhs)

        ic(coeffs.shape, coeffs)

        estimated_rhs = eval_mat @ coeffs

        norm = np.sqrt(((estimated_rhs - rhs)**2).sum(axis=0))
        ic(norm)

        estimated_solution = self.Vandermonde @ self.S @ coeffs

        return estimated_rhs, estimated_solution

class NaturalSpline(AbstractSpline):
    def __init__(self, N=3):
        super().__init__(N, 4)

    def gen_constraint(self, midderivs_free=False):
        N = self.N
        D = self.degree
        constraint = scipy.sparse.dok_array((D*N, D*N))
        row = 0

        """match f(x) to zero at startpoint"""
        constraint[row, 0:D] = [1, -1/2, 1/4, -1/8]
        row += 1

        """match f(x) between interior points"""
        for i in range(N-1):
            constraint[row, D*i:D*(i+2)] = [1, 1/2, 1/4, 1/8] + [-1, 1/2, -1/4, 1/8]
            row += 1

        """match f(x) to zero at endpoint"""
        constraint[row, D*(N-1):D*N] = [1, 1/2, 1/4, 1/8]
        row += 1

        assert not midderivs_free or N % 2 == 0

        """match f'(x) between splines"""
        for i in range(N-1):
            if not midderivs_free or i+1 != N//2:
                constraint[row, D*i:D*(i+2)] = [0, 1, 1, 3/4] + [0, -1, 1, -3/4]
                row += 1    

        """match f''(x) between splines"""
        for i in range(N-1):
            if not midderivs_free or i+1 != N//2:
                constraint[row, D*i:D*(i+2)] = [0, 0, 2, 3] +  [0, 0, -2, 3]
                row += 1

        constraint.resize((row, D*N))
        constraint = constraint.toarray()
        ic(constraint.shape)

        self.constraint = constraint
        
        self.S = scipy.linalg.null_space(self.constraint, rcond=1e-8)
        ic(self.S.shape)
        ic(scipy.linalg.svd(self.S)[1])


class PiecewiseQuadratic(AbstractSpline):
    def __init__(self, N=3):
        super().__init__(N, 3)

    def gen_constraint(self, midderivs_free=False):
        N = self.N
        D = self.degree
        constraint = scipy.sparse.dok_array((D*N, D*N))
        row = 0

        """match f(x) to zero at startpoint"""
        constraint[row, 0:D] = [1, -1/2, 1/4]
        row += 1

        """match f(x) between interior points"""
        for i in range(N-1):
            constraint[row, D*i:D*(i+2)] = [1, 1/2, 1/4] + [-1, 1/2, -1/4]
            row += 1

        """match f(x) to zero at endpoint"""
        constraint[row, D*(N-1):D*N] = [1, 1/2, 1/4]
        row += 1

        assert not midderivs_free or N % 2 == 0

        """match f'(x) between splines"""
        for i in range(N-1):
            if not midderivs_free or i+1 != N//2:
                constraint[row, D*i:D*(i+2)] = [0, 1, 1] + [0, -1, 1]
                row += 1    

        constraint.resize((row, D*N))
        constraint = constraint.toarray()
        ic(constraint.shape)

        self.constraint = constraint
        
        self.S = scipy.linalg.null_space(self.constraint, rcond=1e-8)
        ic(self.S.shape)
        ic(scipy.linalg.svd(self.S)[1])


class PiecewiseLinear(AbstractSpline):
    def __init__(self, N=3):
        super().__init__(N, 2)

    def gen_constraint(self, midderivs_free=False):
        N = self.N
        constraint = scipy.sparse.dok_array((2*N, 2*N))
        row = 0

        """match f(x) to zero at startpoint"""
        constraint[row, 0:2] = [1, -1/2]
        row += 1

        """match f(x) between interior points"""
        for i in range(N-1):
            constraint[row, 2*i:2*(i+2)] = [1, 1/2] + [-1, 1/2]
            row += 1

        """match f(x) to zero at endpoint"""
        constraint[row, 2*(N-1):2*N] = [1, 1/2]
        row += 1

        constraint.resize((row, 2*N))
        constraint = constraint.toarray()
        ic(constraint.shape)

        self.constraint = constraint
        
        self.S = scipy.linalg.null_space(self.constraint, rcond=1e-8)
        ic(self.S.shape)
        ic(scipy.linalg.svd(self.S)[1])

class PiecewiseConstant(AbstractSpline):
    def __init__(self, N=3):
        super().__init__(N, 1)

    def gen_constraint(self, midderivs_free=False):
        N = self.N
        constraint = scipy.sparse.dok_array((N, N))
        row = 0

        """match f(x) to zero at startpoint"""
        constraint[row, 0:1] = [1]
        row += 1

        """match f(x) to zero at endpoint"""
        constraint[row, (N-1):N] = [1]
        row += 1

        """match f(x) match at midpoint"""
        if midderivs_free: 
            mid = N//2-1
            constraint[row, mid:mid+2] = [1] + [-1]
            row += 1

        constraint.resize((row, N))
        constraint = constraint.toarray()
        ic(constraint.shape)

        self.constraint = constraint
        
        self.S = scipy.linalg.null_space(self.constraint, rcond=1e-8)
        ic(self.S.shape)
        ic(scipy.linalg.svd(self.S)[1])

def SplineFactory(tag):
    if tag == 'natural':
        return NaturalSpline
    elif tag == 'piecewise_quadratic':
        return PiecewiseQuadratic
    elif tag == 'piecewise_linear':
        return PiecewiseLinear
    elif tag == 'piecewise_constant':
        return PiecewiseConstant
    else:
        assert False # pragma: no cover

def fit_test_case(*, tag='natural', N=3, nsegs=11):

    spl = SplineFactory(tag)(N)

    spl.gen_constraint()
    spl.gen_Vandermonde(nsegs=nsegs)

    #rhs = np.sin(np.pi/N*spl.xs)
    rhs = (lambda x: x-x**10)(spl.xs/N)
    exact_solution = rhs
    eval_mat = spl.Vandermonde @ spl.S

    estimated_rhs, estimated_solution = spl.backend(eval_mat=eval_mat, rhs=rhs)
    return spl.xs, rhs, exact_solution, estimated_rhs, estimated_solution

def solve_test_case(*, tag='natural', N=3, nsegs=11):

    spl = SplineFactory(tag)(N)

    spl.gen_constraint()
    spl.gen_Vandermonde(nsegs=nsegs)

    spl.gen_deriv_ops()
    rhs = (lambda x: 4*x**2 + 1)(spl.xs/N)
    # solution (should figure out why we need the factor of N*N)
    exact_solution = (lambda x: N*N*(5/6*x - x**2/2 - x**4/3))(spl.xs/N)
    ic(spl.Vandermonde.shape, spl.deriv2_op.shape, spl.S.shape)
    eval_mat = spl.Vandermonde @ - spl.deriv2_op @ spl.S

    estimated_rhs, estimated_solution = spl.backend(eval_mat=eval_mat, rhs=rhs)
    return spl.xs, rhs, exact_solution, estimated_rhs, estimated_solution

def vector_test_case(*, tag='natural', N=4, nsegs=11):

    spl = SplineFactory(tag)(N)

    spl.gen_constraint(midderivs_free=True)
    spl.gen_Vandermonde(nsegs=nsegs)

    G = scipy.sparse.dok_array((spl.xs.shape[0],spl.xs.shape[0]))
    for i in range(spl.xs.shape[0]):
        if i+1 < spl.xs.shape[0]:
            G[i,i] += 1
            G[i+1,i+1] += 1
            G[i,i+1] -= 1
            G[i+1,i] -= 1

    G[0,0] += 1
    G[-1,-1] += 1

    G = G.tocsc()

    rhs = np.zeros(spl.xs.shape)

    rhs[0] = -1/2
    rhs[rhs.shape[0]//2] = 1
    rhs[-1] = -1/2

    ic(G.todok())
    lu = scipy.linalg.lu_factor(G.toarray())
    exact_solution = scipy.linalg.lu_solve(lu, rhs)
    ic(exact_solution)

    eval_mat = G @ spl.Vandermonde @ spl.S

    estimated_rhs, estimated_solution = spl.backend(eval_mat=eval_mat, rhs=rhs)
    return spl.xs, rhs, exact_solution, estimated_rhs, estimated_solution
