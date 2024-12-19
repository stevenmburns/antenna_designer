import numpy as np
from icecream import ic

def gen_bspline(N, xs, x):

    B0 = np.empty(shape=(x.shape[0],N))
    for j in range(N):
        B0[:, j] = np.logical_and(xs[j] <= x, x < xs[j+1])

    ic(B0.shape)

    def aux(B, k):
        lhs = (x[:, np.newaxis] - xs[np.newaxis, :-(k+1)]) / (xs[k:-1] - xs[:-(k+1)])[np.newaxis, :]

        rhs = (xs[np.newaxis, (k+1):] - x[:, np.newaxis]) / (xs[(k+1):] - xs[1:-k])[np.newaxis, :]

        BB = lhs * B[:,:-1] + rhs * B[:, 1:]

        ic(BB.shape)
        
        return BB

    B1 = aux(B0, 1)
    ic(B1.shape)

    B2 = aux(B1, 2)
    ic(B2.shape)

    B3 = aux(B2, 3)
    ic(B3.shape)

    return B0, B1, B2, B3
