
from matplotlib import pyplot as plt

from antenna_designer.spline import gen_matrix


fn = None
#fn = '/dev/null'

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
