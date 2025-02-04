
import numpy as np
from antenna_designer.bspline import gen_bspline, fit_bspline
from matplotlib import pyplot as plt
from icecream import ic

def test_bspline():

    KK = 4

    N = 8

    xs = np.linspace(-3, N+3, N+7)
    x = np.linspace(0, N, 10*N+1)

    fig, ax = plt.subplots(KK)

    for K in range(1, KK+1):
        B = gen_bspline(xs, x, K=K)

        for j in range(B.shape[1]):
            ax[K-1].plot(x, B[:,j], label=f'B{K}_{j}')
        #ax[K-1].legend()

    plt.show()

def test_sparsity():

    N = 8

    xs = np.linspace(-3, N+3, N+7)
    x = np.linspace(0, N, 10*N+1)

    B = gen_bspline(xs, x)    

    plt.spy(B)
    plt.gca().set_aspect('equal')

    plt.show()

def test_fit():
    fig, ax = plt.subplots(1)

    x = np.linspace(0, 1, 401)
    y = np.sin(2*np.pi*x)
    ax.plot(x, y, label='actual y')

    for N in range(1,4):

        xs = np.linspace(-3, N+3, N+7)

        coeffs, norm, B = fit_bspline(xs, x*N, y)
        ic(N, coeffs.shape, coeffs, norm)

        predicted_y = B @ coeffs

        ax.plot(x, predicted_y, label=f'predicted y {N}')

    ax.legend()
    plt.show()
    
    
