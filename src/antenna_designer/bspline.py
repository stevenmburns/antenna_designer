import numpy as np
from icecream import ic

def gen_bspline(N, xs, x):

    B0 = np.empty(shape=(x.shape[0],N))
    B00 = np.logical_and(x[:, np.newaxis] >= xs[np.newaxis, :-1],
                         x[:, np.newaxis] <  xs[np.newaxis, 1:])



    for j in range(N):
        B0[:, j] = np.logical_and(xs[j] <= x, x < xs[j+1])

    assert (B00 == B0).all()

    def aux(B, k):
        lhs = (x[:, np.newaxis] - xs[np.newaxis, :-(k+1)]) / (xs[k:-1] - xs[:-(k+1)])[np.newaxis, :]

        rhs = (xs[np.newaxis, (k+1):] - x[:, np.newaxis]) / (xs[(k+1):] - xs[1:-k])[np.newaxis, :]

        BB = lhs * B[:,:-1] + rhs * B[:, 1:]

        return BB

    Bs = [B0]
    for i in range(1, 4):
        Bs.append(aux(Bs[-1], i))

    return tuple(Bs)
