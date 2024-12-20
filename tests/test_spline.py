import pytest
from matplotlib import pyplot as plt

from antenna_designer.spline import fit_test_case, solve_test_case, vector_test_case


fn = None
#fn = '/dev/null'

spline_models = ('natural', 'piecewise_quadratic', 'piecewise_linear', 'piecewise_constant')

@pytest.mark.parametrize("tag", spline_models)
def test_solve(tag):

    N = 3
    nsegs = 11

    fig, (ax0, ax1)  = plt.subplots(1, 2)

    for NN in range(N, N+1, 2):
        xs, rhs, exact_solution, estimated_rhs, estimated_solution = solve_test_case(tag=tag, N=NN, nsegs=nsegs)
        ax1.plot(xs/N, rhs, label='known rhs')
        ax1.plot(xs/NN, estimated_rhs, label=f'{NN} estimated rhs')

        ax0.plot(xs/N, exact_solution, label='exact solution')
        ax0.plot(xs/NN, estimated_solution, label=f'{NN} estimated solution')

    fig.suptitle(f'solve {tag} N={N} nsegs={nsegs}')
    ax0.legend(loc='upper right')
    ax1.legend(loc='upper left')
    plt.show()

@pytest.mark.parametrize("tag", spline_models)
def test_fit(tag):

    N = 3
    nsegs = 11

    fig, ax  = plt.subplots(1, 1)

    for NN in range(N, N+1, 2):
        xs, rhs, _, estimated_rhs, _ = fit_test_case(tag=tag, N=NN, nsegs=nsegs)
        ax.plot(xs/NN, rhs, label=f'{NN} known rhs')
        ax.plot(xs/NN, estimated_rhs, label=f'{NN} estimated rhs')

    fig.suptitle(f'fit {tag} N={N} nsegs={nsegs}')
    plt.legend()
    plt.show()

@pytest.mark.parametrize("tag", spline_models)
def test_vector(tag):
    N = 12
    nsegs = 10

    fig, (ax0, ax1)  = plt.subplots(1, 2)

    for NN in range(N, N+1, 2):
        xs, rhs, exact_solution, estimated_rhs, estimated_solution = vector_test_case(tag=tag, N=NN, nsegs=nsegs)

        ax1.plot(xs/NN, rhs, label=f'{NN} driven current')
        ax1.plot(xs/NN, estimated_rhs, label=f'{NN} estimated current')

        ax0.plot(xs/NN, exact_solution, label=f'{NN} expected voltage')
        ax0.plot(xs/NN, estimated_solution, label=f'{NN} voltage')


    fig.suptitle(f'vector {tag} N={N} nsegs={nsegs}')
    ax0.legend(loc='upper right')
    ax1.legend(loc='upper left')
    plt.show()
