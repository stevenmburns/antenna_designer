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

    lhs = (x[:, np.newaxis] - xs[np.newaxis, :-2]) / (xs[1:-1] - xs[:-2])[np.newaxis, :]

    rhs = (xs[np.newaxis, 2:] - x[:, np.newaxis]) / (xs[2:] - xs[1:-1])[np.newaxis, :]

    B1 = lhs * B0[:,:-1] + rhs * B0[:, 1:]

    assert (B1 == aux(B0, 1)).all()

    ic(B1.shape)

    lhs = (x[:, np.newaxis] - xs[np.newaxis, :-3]) / (xs[2:-1] - xs[:-3])[np.newaxis, :]

    rhs = (xs[np.newaxis, 3:] - x[:, np.newaxis]) / (xs[3:] - xs[1:-2])[np.newaxis, :]

    B2 = lhs * B1[:,:-1] + rhs * B1[:, 1:]

    assert (B2 == aux(B1, 2)).all()

    ic(B2.shape)

    lhs = (x[:, np.newaxis] - xs[np.newaxis, :-4]) / (xs[3:-1] - xs[:-4])[np.newaxis, :]

    rhs = (xs[np.newaxis, 4:] - x[:, np.newaxis]) / (xs[4:] - xs[1:-3])[np.newaxis, :]

    B3 = lhs * B2[:,:-1] + rhs * B2[:, 1:]

    assert (B3 == aux(B2, 3)).all()

    ic(B3.shape)

    return B0, B1, B2, B3
