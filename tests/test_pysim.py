import pytest
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import time

from antenna_designer.pysim import PySim
from antenna_designer.augmented_spline_pysim import AugmentedSplinePySim

from antenna_designer.core import save_or_show
from antenna_designer.pysim_accelerators import dist_outer_product

from matplotlib import pyplot as plt
import numpy as np
import scipy
from icecream import ic

import skrf

fn = None
#fn = '/dev/null'

def test_extension():
    nsegs = 20000
    pts = np.array([[0,0,z] for z in range(nsegs+1)])/(2*nsegs)

    t = time.time()
    for i in range(10):
        _ = dist_outer_product(pts, pts)
    ic('dist_outer_product', time.time()-t)


def test_impedance_nsegs():
    xs = [21, 41, 61, 81, 101, 201, 401, 801]
    xs = np.array(xs)
    z0 = 50



    fig, ax0 = plt.subplots()
    skrf.plotting.smith(draw_labels=True, chart_type='z')


    #plt.plot(xs, np.abs(zs), marker='s')
    #plt.plot(xs, np.imag(zs), marker='s')

    for ntrap, color in [(0,'tab:green'),(2,'tab:blue'),(8,'tab:red'),(16,'tab:purple')]:

        zs = []
        for nsegs in xs:
            z, _ = PySim(nsegs=nsegs).compute_impedance(ntrap=ntrap)
            ic(nsegs,z)
            zs.append(z)

        zs = np.array(zs)

        normalized_zs = zs/z0
        reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
        skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z', marker='s', linestyle='None')

    save_or_show(plt, fn)

def test_spline_impedance_nsegs():
    xs = [21] # , 41, 61, 81, 101, 201, 401, 801]
    xs = np.array(xs)
    z0 = 50

    fig, ax0 = plt.subplots()
    skrf.plotting.smith(draw_labels=True, chart_type='z')


    #plt.plot(xs, np.abs(zs), marker='s')
    #plt.plot(xs, np.imag(zs), marker='s')

    for ntrap, color in [
            (2,'tab:blue'),
            (4,'tab:green'),
#            (8,'tab:red'),
#            (16,'tab:purple'),
    ]:
        zs = []
        for nsegs in xs:
            z, _ = AugmentedSplinePySim(nsegs=nsegs).compute_impedance(ntrap=ntrap)
            ic(nsegs,z)
            zs.append(z)

        zs = np.array(zs)

        normalized_zs = zs/z0
        reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
        skrf.plotting.plot_smith(reflection_coefficients, color=color, label=f'nsegs: {nsegs} ntrap: {ntrap}', draw_labels=True, chart_type='z', marker='s', linestyle='None')

    plt.legend()
    save_or_show(plt, fn)


def test_spline_fit():
    nsegs=1001
    nsample = 100
    assert (nsegs-1) % nsample == 0

    halfdriver_factors = np.linspace(.9,1,6)

    fig, ax0 = plt.subplots()
    ax1 = ax0.twinx()

    for halfdriver_factor in halfdriver_factors:

        _, i = PySim(nsegs=nsegs, halfdriver_factor=halfdriver_factor).compute_impedance(ntrap=16)

        xs = np.linspace(0,nsegs-1,nsegs)

        i_sample = i[::nsample]
        xs_sample = xs[::nsample]

        interp = scipy.interpolate.CubicSpline(xs_sample, i_sample)

        color = 'tab:blue'
        ax0.plot(xs, np.abs(i), color=color)
        color = 'tab:purple'
        ax0.plot(xs, np.abs(interp(xs)), color=color)


        color = 'tab:green'
        ax1.plot(xs, np.angle(i)*180/np.pi, color=color)
        color = 'tab:olive'
        ax1.plot(xs, np.angle(interp(xs))*180/np.pi, color=color)
    plt.show()



def test_svd_currents_nsmallest():

    nsegs=101

    _, (i, i_svd_all) = PySim(nsegs=nsegs, run_svd=True).compute_impedance(ntrap=0)

    color = 'tab:blue'
    plt.plot(np.abs(i), color=color)

    color = 'tab:red'
    plt.plot(np.abs(i_svd_all), color=color)

    for nsmallest, color in [(1,'tab:green'), (2,'tab:purple')]:
        _, (_, i_svd) = PySim(nsegs=nsegs,nsmallest=nsmallest, run_svd=True).compute_impedance(ntrap=0)
        ic(nsmallest, np.linalg.norm(i_svd-i_svd_all))
        plt.plot(np.abs(i_svd), color=color)

    save_or_show(plt, fn)

def test_spline_currents():

    fig, ax = plt.subplots(2, 2)

    nsegs = 401

    for N, color in [
            (4,'tab:green'),
            (6,'tab:purple'),
#            (40,'tab:blue'),
#            (80,'tab:orange'),
    ]:
        _, (i, orig_i, matched_i) = AugmentedSplinePySim(nsegs=nsegs).compute_impedance(ntrap=2, N=N)

        ax[0][0].plot(np.abs(i), color=color, label=f'{N}' )
        ax[0][1].plot(np.angle(i)*180/np.pi, color=color, label=f'{N}')


        ax[1][0].plot(np.abs(orig_i), color=color, label=f'{N} orig' )
        ax[1][1].plot(np.angle(orig_i)*180/np.pi, color=color, label=f'{N} orig')

        ax[1][0].plot(np.abs(matched_i), color=color, label=f'{N} matched' )
        ax[1][1].plot(np.angle(matched_i)*180/np.pi, color=color, label=f'{N} matched')


    ax[0][0].legend()
    ax[0][1].legend()
    ax[1][0].legend()
    ax[1][1].legend()
    save_or_show(plt, fn)


def test_iterative_improvement():
    PySim(nsegs=401, run_iterative_improvement=True).compute_impedance(ntrap=0)


def test_sweep_halfdriver():

    nsegs=401
    z0 = 50

    fig, ax0 = plt.subplots()
    skrf.plotting.smith(draw_labels=True, chart_type='z')

    xs = np.linspace(.9,1,21)

    for ntrap, color in ((0,'tab:green'),(4,'tab:blue'),(16,'tab:purple')):

        t = time.time()
        zs = []
        for x in xs:
            z, _ = PySim(halfdriver_factor=x,nsegs=nsegs).compute_impedance(ntrap=4)
            zs.append(z)
        print('augmented ntrap=4', time.time()-t)
        zs = np.array(zs)

        normalized_zs = zs/z0
        reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
        skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z', marker='s', linestyle='None')

    save_or_show(plt, fn)

nsegs = 801
nrepeat = 1
ntrap = 8

@pytest.mark.parametrize('engine,ntrap', [('python', 0), ('python', ntrap), ('accelerated', ntrap), ('test', ntrap)])
def test_param(engine, ntrap):
    ps = PySim(nsegs=nsegs)

    t = time.time()
    for i in range(nrepeat):
        z, i = ps.compute_impedance(ntrap=ntrap, engine=engine)

    ic(f'engine {engine}', time.time()-t)
