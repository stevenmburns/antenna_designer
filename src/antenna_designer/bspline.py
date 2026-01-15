import numpy as np
from icecream import ic

from .spline import AbstractSpline


def gen_bspline(xs, x, *, K=4):

    B0 = np.logical_and(x[:, np.newaxis] >= xs[np.newaxis, :-1],
                         x[:, np.newaxis] <  xs[np.newaxis, 1:])

    def aux(B, k):
        lhs = (x[:, np.newaxis] - xs[np.newaxis, :-(k+1)]) / (xs[k:-1] - xs[:-(k+1)])[np.newaxis, :]

        rhs = (xs[np.newaxis, (k+1):] - x[:, np.newaxis]) / (xs[(k+1):] - xs[1:-k])[np.newaxis, :]

        BB = lhs * B[:,:-1] + rhs * B[:, 1:]

        return BB

    B = B0
    for i in range(1, K):
        B = aux(B, i)

    ic(B)

    return B

def fit_bspline(xs, x, y):
    B = gen_bspline(xs, x)
    coeffs = AbstractSpline.pseudo_solve(B, y)

    return coeffs, np.linalg.norm(y- B @ coeffs), B
