import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import time

from antenna_designer import pysim
from antenna_designer.core import save_or_show
from antenna_designer.pysim_accelerators import dist_outer_product

from matplotlib import pyplot as plt
import numpy as np
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
            z, _ = pysim.PySim(nsegs=nsegs).augmented_compute_impedance(ntrap=ntrap)
            ic(nsegs,z)
            zs.append(z)

        zs = np.array(zs)

        normalized_zs = zs/z0
        reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
        skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z', marker='s', linestyle='None')

    save_or_show(plt, fn)


def test_svd_currents_nsmallest():

    nsegs=101

    _, (i, i_svd_all) = pysim.PySim(nsegs=nsegs, run_svd=True).augmented_compute_impedance(ntrap=0)

    color = 'tab:blue'
    plt.plot(np.abs(i), color=color)

    color = 'tab:red'
    plt.plot(np.abs(i_svd_all), color=color)

    for nsmallest, color in [(1,'tab:green'), (2,'tab:purple')]:
        _, (_, i_svd) = pysim.PySim(nsegs=nsegs,nsmallest=nsmallest, run_svd=True).augmented_compute_impedance(ntrap=0)
        ic(nsmallest, np.linalg.norm(i_svd-i_svd_all))
        plt.plot(np.abs(i_svd), color=color)

    save_or_show(plt, fn)


def test_iterative_improvement():
    pysim.PySim(nsegs=401, run_iterative_improvement=True).augmented_compute_impedance(ntrap=0)


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
            z, _ = pysim.PySim(halfdriver_factor=x,nsegs=nsegs).augmented_compute_impedance(ntrap=4)
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

def test_augmented_python_ntrap0():
    ps = pysim.PySim(nsegs=nsegs)

    t = time.time()
    for i in range(nrepeat):
        z, i = ps.augmented_compute_impedance(ntrap=0, engine='python')
    ic('augmented python ntrap=0', time.time()-t)

def test_augmented():
    ps = pysim.PySim(nsegs=nsegs)

    t = time.time()
    for i in range(nrepeat):
        z, i = ps.augmented_compute_impedance(ntrap=ntrap, engine='accelerated')
    ic('augmented accelerated', time.time()-t)

def test_augmented_python():
    ps = pysim.PySim(nsegs=nsegs)

    t = time.time()
    for i in range(nrepeat):
        z, i = ps.augmented_compute_impedance(ntrap=ntrap, engine='python')
    ic('augmented python', time.time()-t)

def test_augmented_test():
    ps = pysim.PySim(nsegs=nsegs)

    t = time.time()
    for i in range(nrepeat):
        z, i = ps.augmented_compute_impedance(ntrap=ntrap, engine='test')
    ic('augmented test', time.time()-t)
