from . import Antenna
from .core import save_or_show

import numpy as np

import matplotlib.pyplot as plt

def get_pattern_rings(antenna_builder):
  a = Antenna(antenna_builder)
  a.set_freq_and_execute()

  del_theta = 1
  del_phi = 1
  n_theta = 90
  n_phi = 360

  assert 90 % n_theta == 0 and 90 == del_theta * n_theta
  assert 360 % n_phi == 0 and 360 == del_phi * n_phi


  a.c.rp_card(0, n_theta, n_phi+1, 0, 5, 0, 0, 0, 0, del_theta, del_phi, 0, 0)

  thetas = np.linspace(0,90-del_theta,n_theta)
  phis = np.linspace(0,360,n_phi+1)

  rings = []

  for theta_index, theta in enumerate(thetas):
    ring = [a.c.get_gain(0, theta_index, phi_index) for phi_index, phi in enumerate(phis)]
    rings.append(ring)
             
  max_gain = a.c.get_gain_max(0)
  min_gain = a.c.get_gain_min(0)

  del a

  return rings, max_gain, min_gain, thetas, phis

def get_elevation(a):
  del_theta = 1
  del_phi = 360
  n_theta = 90
  n_phi = 1

  assert 90 % n_theta == 0 and 90 == del_theta * n_theta
  assert 360 % n_phi == 0 and 360 == del_phi * n_phi


  a.c.rp_card(0, n_theta, n_phi+1, 0, 5, 0, 0, 0, 0, del_theta, del_phi, 0, 0)

  thetas = np.linspace(0,90-del_theta,n_theta)
  phis = np.linspace(0,360,n_phi+1)

  rings = []

  for theta_index, theta in enumerate(thetas):
    ring = [a.c.get_gain(0, theta_index, phi_index) for phi_index, phi in enumerate(phis)]
    rings.append(ring)
             
  max_gain = a.c.get_gain_max(0)
  min_gain = a.c.get_gain_min(0)

  del a

  return rings, max_gain, min_gain, thetas, phis


def plot_patterns(rings_lst, names, thetas, phis, elevation_angle=15, fn=None):
  fig, axes = plt.subplots(ncols=2, subplot_kw={'projection': 'polar'})

  axes[0].set_rticks([-12, -6, 0, 6, 12])

  for nm, rings in zip(names, rings_lst):
    for theta, ring in list(zip(thetas, rings)):
      if abs(theta-(90-elevation_angle)) < 0.1:
        axes[0].plot(np.deg2rad(phis),ring,marker='',label=f"{(90-theta):.0f} {nm}")

  axes[0].legend(loc="lower left")

  print(len(rings),len(rings[0]))

  n = len(rings_lst[0][0])
  assert (n-1) % 2 == 0

  azimuth_f = 0
  azimuth_r = (n-1)//2

  if False:
    delta_azimuth = 0
    azimuth_f -= delta_azimuth
    asimuth_f %= n-1

    azimuth_r += delta_azimuth
    asimuth_r %= n-1

    assert 0 <= azimuth_f < n-1
    assert 0 <= azimuth_r < n-1

  elevations = [list(reversed([ring[azimuth_f] for ring in rings]))+[ring[azimuth_r] for ring in rings] for rings in rings_lst]
  el_thetas = list(reversed(list(90-thetas))) + list(90+thetas)

  axes[1].set_rticks([-12, -6, 0, 6, 12])

  for elevation in elevations:
    axes[1].plot(np.deg2rad(el_thetas),elevation,marker='')

  save_or_show(plt, fn)


def compare_patterns(antenna_builders, elevation_angle=15, fn=None, builder_names=None):
  if builder_names is None:
    builder_names = ['Unknown' for _ in antenna_builders]

  rings_lst = []

  for antenna_builder in antenna_builders:
    rings, max_gain, min_gain, thetas, phis = get_pattern_rings(antenna_builder)
    rings_lst.append(rings)

  plot_patterns(rings_lst, builder_names, thetas, phis, elevation_angle, fn)


def pattern(antenna_builder, elevation_angle=15, fn=None):

  rings, max_gain, min_gain, thetas, phis = get_pattern_rings(antenna_builder)

  elevation = [ring[0] for ring in rings]

  fig, axes = plt.subplots(ncols=2, subplot_kw={'projection': 'polar'})

  axes[0].set_rticks([-12, -6, 0, 6, 12])

  for theta, ring in list(zip(thetas, rings)):
    if abs(theta-(90-elevation_angle)) < 0.1:
      axes[0].plot(np.deg2rad(phis),ring,marker='',label=f"{(90-theta):.0f}")

  axes[0].legend(loc="lower left")

  n = len(rings[0])
  assert (n-1) % 2 == 0
  elevation = list(reversed([ring[0] for ring in rings]))+[ring[(n-1)//2] for ring in rings]
  el_thetas = list(reversed(list(90-thetas))) + list(90+thetas)

  axes[1].set_rticks([-12, -6, 0, 6, 12])

  axes[1].plot(np.deg2rad(el_thetas),elevation,marker='')

  save_or_show(plt, fn)


def pattern3d(antenna_builder, fn=None):
  a = Antenna(antenna_builder)
  a.set_freq_and_execute()

  del_theta = 3
  del_phi = 6
  n_theta = 30
  n_phi = 60

  assert 90 % n_theta == 0 and 90 == del_theta * n_theta
  assert 360 % n_phi == 0 and 360 == del_phi * n_phi

  a.c.rp_card(0, n_theta, n_phi+1, 0, 5, 0, 0, 0, 0, del_theta, del_phi, 0, 0)

  thetas = np.linspace(0,90-del_theta,n_theta)
  phis = np.linspace(0,360,n_phi+1)

  rhos = [[a.c.get_gain(0, theta_index, phi_index) for theta_index, _ in enumerate(thetas)] for phi_index, _ in enumerate(phis)]
             
  del a

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')


  Theta, Phi = np.meshgrid(np.deg2rad(thetas),np.deg2rad(phis))
  Rho = 10**(np.array(rhos)/10)

  X = Rho * np.sin(Theta)*np.cos(Phi)
  Y = Rho * np.sin(Theta)*np.sin(Phi)
  Z = Rho * np.cos(Theta)

  ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
  ax.set_aspect('equal')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  save_or_show(plt, fn)



