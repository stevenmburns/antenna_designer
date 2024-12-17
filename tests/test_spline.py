import pytest
from matplotlib import pyplot as plt

from antenna_designer.spline import fit_test_case, solve_test_case, vector_test_case


fn = None
#fn = '/dev/null'

spline_models = ('natural', 'piecewise_quadratic', 'piecewise_linear', 'piecewise_constant')

@pytest.mark.parametrize("tag", spline_models)
def test_solve(tag):

    N = 4
    nrepeats = 20

    for NN in range(N, N+1, 2):
        xs, rhs, exact_solution, estimated_rhs, estimated_solution = solve_test_case(tag=tag, N=NN, nrepeats=nrepeats)
        plt.plot(xs/N, rhs, label='known rhs')
        plt.plot(xs/N, exact_solution, label='exact solution')
        plt.plot(xs/NN, estimated_rhs, label=f'{NN} estimated rhs')
        plt.plot(xs/NN, estimated_solution, label=f'{NN} estimated solution')

        coarse_xs = xs[nrepeats//2::nrepeats]
        coarse_new_ys = estimated_rhs[nrepeats//2::nrepeats]
        plt.plot(coarse_xs/NN, coarse_new_ys, marker='s', linestyle='None')


    plt.legend()
    plt.show()

@pytest.mark.parametrize("tag", spline_models)
def test_fit(tag):

    N = 3
    nrepeats = 20

    for NN in range(N, N+1, 2):
        xs, rhs, _, estimated_rhs, _ = fit_test_case(tag=tag, N=NN, nrepeats=nrepeats)
        plt.plot(xs/NN, rhs, label=f'{NN} known rhs')
        plt.plot(xs/NN, estimated_rhs, label=f'{NN} estimated rhs')

        coarse_xs = xs[nrepeats//2::nrepeats]
        coarse_new_ys = estimated_rhs[nrepeats//2::nrepeats]
        plt.plot(coarse_xs/NN, coarse_new_ys, marker='s', linestyle='None')


    plt.legend()
    plt.show()

@pytest.mark.parametrize("tag", spline_models)
def test_vector(tag):
    N = 10
    nrepeats = 20

    fig, (ax0, ax1)  = plt.subplots(1, 2)

    for NN in range(N, N+1, 2):
        xs, rhs, exact_solution, estimated_rhs, estimated_solution = vector_test_case(tag=tag, N=NN, nrepeats=nrepeats)

        ax1.plot(xs/NN, rhs, label=f'{NN} driven current')
        ax1.plot(xs/NN, estimated_rhs, label=f'{NN} estimated current')

        ax0.plot(xs/NN, exact_solution, label=f'{NN} expected voltage')
        ax0.plot(xs/NN, estimated_solution, label=f'{NN} voltage')

        coarse_xs = xs[nrepeats//2::nrepeats]
        coarse_new_ys = estimated_rhs[nrepeats//2::nrepeats]
        ax1.plot(coarse_xs/NN, coarse_new_ys, marker='s', linestyle='None')


    ax0.legend(loc='upper right')
    ax1.legend(loc='upper left')
    plt.show()
