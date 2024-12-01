from . import Antenna
from .core import save_or_show
from .far_field import get_elevation
from icecream import ic

import numpy as np

import matplotlib.pyplot as plt
import skrf
import skrf.plotting

def build_and_get_elevation(antenna_builder):
  a = Antenna(antenna_builder)
  a.set_freq_and_execute()
  return get_elevation(a)

def resolve_range(default_value, rng, center, fraction):
  if rng is None:
    if fraction is None:
      fraction = 1.25

    if center is None:
      center = default_value

    rng = (center / fraction, center * fraction)

  return rng


def sweep_freq(antenna_builder, *, z0=200, rng=None, center=None, fraction=None, npoints=21, fn=None):

  rng = resolve_range(antenna_builder.freq, rng, center, fraction)

  min_freq = rng[0]
  max_freq = rng[1]
  n_freq = npoints-1
  del_freq = (max_freq- min_freq)/n_freq

  xs = np.linspace(min_freq, max_freq, n_freq+1)
  
  a = Antenna(antenna_builder)

  a.c.fr_card(0, n_freq+1, min_freq, del_freq)
  a.c.xq_card(0) # Execute simulation
  
  zs = np.array([a.impedance(freq_index,sweep=True) for freq_index in range(len(xs))])

  del a

  zs = np.array(zs)

  reflection_coefficient = (zs - z0) / (zs + z0)
  rho = np.abs(reflection_coefficient)
  swr = (1+rho)/(1-rho)

  rho_db = np.log10(rho)*10.0

  fig, ax0 = plt.subplots()
  color = 'tab:red'
  ax0.set_xlabel('freq')
  ax0.set_ylabel('rho_db', color=color)
  ax0.tick_params(axis='y', labelcolor=color)
  for i in range(rho_db.shape[1]):
    ax0.plot(xs, rho_db[:,i], color=color)


  color = 'tab:blue'
  ax1 = ax0.twinx()
  ax1.set_ylabel('swr', color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  for i in range(swr.shape[1]):
    ax1.plot(xs, swr[:,i], color=color)

  fig.tight_layout()

  save_or_show(plt, fn)


def sweep_gain(antenna_builder, nm, *, rng=None, center=None, fraction=None, npoints=21, fn=None):

  rng = resolve_range(getattr(antenna_builder, nm), rng, center, fraction)

  xs = np.linspace(rng[0],rng[1],npoints)

  gs = []
  for x in xs:
    setattr(antenna_builder, nm, x)
    _, max_gain, _, _, _ = build_and_get_elevation(antenna_builder)
    gs.append(max_gain)

  gs = np.array(gs)
  
  fig, ax0 = plt.subplots()
  color = 'tab:red'
  ax0.set_xlabel(nm)
  ax0.set_ylabel('max_gain', color=color)
  ax0.tick_params(axis='y', labelcolor=color)
  ax0.plot(xs, gs, color=color)

  save_or_show(plt, fn)

def sweep(antenna_builder, nm, *, rng=None, center=None, fraction=None, npoints=21, use_smithchart=False, z0=50, markers=[], fn=None):

  rng = resolve_range(getattr(antenna_builder, nm), rng, center, fraction)

  xs = np.linspace(rng[0],rng[1],npoints)

  zs = []
  for x in xs:
    setattr(antenna_builder, nm, x)
    zs.append(Antenna(antenna_builder).impedance())

  marker_zs = []
  for x in markers:
    setattr(antenna_builder, nm, x)
    marker_zs.append(Antenna(antenna_builder).impedance())

  zs = np.array(zs)
  marker_xs = np.array(markers)
  marker_zs = np.array(marker_zs)

  nwidth = zs.shape[1] if npoints > 0 else marker_zs.shape[1]
  ic(nwidth, npoints, markers, zs.shape, marker_zs.shape)

  if use_smithchart:
    fig, ax0 = plt.subplots()
    color = 'tab:red'
    skrf.plotting.smith(draw_labels=True, chart_type='z')
    for i in range(nwidth):
      if zs.shape[0] > 0:
        normalized_zs = zs/z0
        reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
        skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z')

      if marker_zs.shape[0] > 0:
        normalized_zs = marker_zs/z0
        reflection_coefficients = (normalized_zs-1)/(normalized_zs+1)
        skrf.plotting.plot_smith(reflection_coefficients, color=color, draw_labels=True, chart_type='z', marker='s', linestyle='None')
      
  else:
    fig, ax0 = plt.subplots()
    color = 'tab:red'
    ax0.set_ylabel('z real', color=color)
    ax0.tick_params(axis='y', labelcolor=color)
    for i in range(nwidth):
      if zs.shape[0] > 0:
        ax0.plot(xs, np.real(zs)[:,i], color=color)
      if marker_zs.shape[0] > 0:
        ax0.plot(marker_xs, np.real(marker_zs)[:,i], color=color, marker='s', linestyle='None')

    color = 'tab:blue'
    ax1 = ax0.twinx()
    ax1.set_ylabel('z imag', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    for i in range(nwidth):
      if zs.shape[0] > 0:
        ax1.plot(xs, np.imag(zs)[:,i], color=color)
      if marker_zs.shape[0] > 0:
        ax1.plot(marker_xs, np.imag(marker_zs)[:,i], color=color, marker='s', linestyle='None')


    fig.tight_layout()

  save_or_show(plt, fn)



