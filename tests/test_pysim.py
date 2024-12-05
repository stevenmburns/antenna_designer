import time

from antenna_designer import pysim
from antenna_designer.core import save_or_show
from matplotlib import pyplot as plt
import numpy as np
from icecream import ic

import skrf

#fn = None
fn = '/dev/null'

def test_impedance_nsegs():
    xs = [11, 21, 41, 101, 201, 401, 801]
    zs = []
    for nsegs in xs:
        z, _ = pysim.PySim(nsegs=nsegs).vectorized_compute_impedance()
        ic(nsegs,z)
        zs.append(z)

    xs = np.array(xs)
    zs = np.array(zs)

    plt.plot(xs, np.abs(zs), marker='s')
    plt.plot(xs, np.imag(zs), marker='s')

    save_or_show(plt, fn)


def test_svd_currents_nsmallest():

    nsegs=101

    _, (i, i_svd_all) = pysim.PySim(nsegs=nsegs).vectorized_compute_impedance()

    color = 'tab:blue'
    plt.plot(np.abs(i), color=color)

    color = 'tab:red'
    plt.plot(np.abs(i_svd_all), color=color)

#    for rcond in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    color = 'tab:green'
    for nsmallest in [1, 2, 3]:
        _, (_, i_svd) = pysim.PySim(nsegs=nsegs,nsmallest=nsmallest).vectorized_compute_impedance()
        ic(nsmallest, np.linalg.norm(i_svd-i_svd_all))
        plt.plot(np.abs(i_svd), color=color)

    save_or_show(plt, fn)


def test_sweep_halfdriver():

    nsegs=1001

    xs = np.linspace(.9,1,2)

    # run once for jit
    pysim.PySim(nsegs=nsegs).jax_compute_impedance()

    t = time.time()
    zas = []
    for x in xs:
        z, _ = pysim.PySim(halfdriver_factor=x,nsegs=nsegs).jax_compute_impedance()
        zas.append(z)
    print('jax', time.time()-t)
    zas = np.array(zas)

    t = time.time()
    zbs = []
    for x in xs:
        z, _ = pysim.PySim(halfdriver_factor=x,nsegs=nsegs).vectorized_compute_impedance()
        zbs.append(z)
    print('vectorized', time.time()-t)
    zbs = np.array(zbs)

    if False:
        t = time.time()
        zcs = []
        for x in xs:
            z, _ = pysim.PySim(halfdriver_factor=x,nsegs=nsegs).compute_impedance()
            zcs.append(z)
        print('slow', time.time()-t)
        zcs = np.array(zcs)

    z0 = 50

    fig, ax0 = plt.subplots()
    skrf.plotting.smith(draw_labels=True, chart_type='z')

    normalized_zs = zas/z0
    color = 'tab:red'
    reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
    skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z', marker='s', linestyle='None')

    normalized_zs = zbs/z0
    color = 'tab:blue'
    reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
    skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z', marker='s', linestyle='None')

    if False:
        normalized_zs = zcs/z0
        color = 'tab:green'
        reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
        skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z', marker='s', linestyle='None')

    save_or_show(plt, fn)

def test_slow():
    ps = pysim.PySim()
    z, i = ps.compute_impedance()

def test_vectorized():
    ps = pysim.PySim()
    z, i = ps.vectorized_compute_impedance()
