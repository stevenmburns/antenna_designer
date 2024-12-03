import time

from antenna_designer import pysim
from matplotlib import pyplot as plt
import numpy as np

import skrf

def test_sweep_halfdriver():

    xs = np.linspace(.9,1,11)

    t = time.time()
    zas = []
    for x in xs:
        za, _ = pysim.vectorized_compute_impedance(halfdriver_factor=x)
        zas.append(za)
    print('vectorized', time.time()-t)

    t = time.time()
    zbs = []
    for x in xs:
        zb, _ = pysim.compute_impedance(halfdriver_factor=x)
        zbs.append(zb)
    print('slow', time.time()-t)

    zas = np.array(zas)
    zbs = np.array(zbs)

    z0 = 50

    fig, ax0 = plt.subplots()
    skrf.plotting.smith(draw_labels=True, chart_type='z')

    normalized_zs = zas/z0
    color = 'tab:red'
    reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
    skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z')# , marker='s', linestyle='None')

    normalized_zs = zbs/z0
    color = 'tab:blue'
    reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
    skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z')# , marker='s', linestyle='None')

    plt.show()


def test_vectorized():
    z, i = pysim.vectorized_compute_impedance()
