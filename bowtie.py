#
#  Simple vertical monopole antenna simulation using python-necpp
#  pip install necpp
#
from necpp import *
import math
import numpy as np

from scipy.optimize import minimize_scalar, minimize

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d

def handle_nec(result):
  if (result != 0):
    print(nec_error_message())

def draw(tups):

  pairs = [(p0,p1) for p0, p1, _ in tups]

  lc = LineCollection(pairs, colors=(1, 0, 0, 1), linewidths=1)

  fig, ax = plt.subplots()
  ax.add_collection(lc)
  ax.autoscale(axis='x')
  ax.set_aspect('equal')
  ax.margins(0.1)
  plt.show()

  exit()


def geometry(freq, slope, base, length):
  
  conductivity = 5.8e7 # Copper
  ground_conductivity = 0.002
  ground_dielectric = 10

  wavelength = 3e8/(1e6*freq)
  eps = 0.05

  # diag = sqrt(x^2 + (x*slope)^2) = x*sqrt(1+slope^2)
  # length/2 = diag + x*slope = x*(slope + sqrt(1+slope^2))

  x = 0.5*length/(slope + math.sqrt(1+slope**2))
  z = slope*x

  n_seg0 = 51
  n_seg1 = 3

  tups = []
  tups.extend([((-x,   base),     (-x,   base+z),   n_seg0)])
  tups.extend([((-x,   base+z),   (-eps, base+eps), n_seg0)])
  tups.extend([((-eps, base+eps), ( eps, base+eps), n_seg1)])
  tups.extend([(( eps, base+eps), ( x,   base+z),   n_seg0)])
  tups.extend([(( x,   base+z),   ( x,   base),     n_seg0)])
  tups.extend([((-x, base),       (-x, base-z),     n_seg0)])
  tups.extend([((-x, base-z),     (-eps, base-eps), n_seg0)])
  tups.extend([(( eps, base-eps), ( x, base-z),     n_seg0)])
  tups.extend([(( x, base-z),     ( x, base),       n_seg0)])
  tups.extend([((-eps, base-eps), ( eps, base-eps), n_seg1)])

  #draw(tups)

  nec = nec_create()

  for idx, (p0, p1, n_seg) in enumerate(tups, start=1):
    handle_nec(nec_wire(nec, idx, n_seg, 0, p0[0], p0[1], 0, p1[0], p1[1], 0.002, 1.0, 1.0))

  handle_nec(nec_geometry_complete(nec, 1))
  handle_nec(nec_ld_card(nec, 5, 0, 0, 0, conductivity, 0.0, 0.0))
  handle_nec(nec_gn_card(nec, 0, 0, ground_dielectric, ground_conductivity, 0, 0, 0, 0))
  handle_nec(nec_fr_card(nec, 0, 1, freq, 0))
  handle_nec(nec_ex_card(nec, 0, len(tups), (n_seg1+1)//2, 0, 1.0, 0, 0, 0, 0, 0)) 

  return nec

def pattern():
  freq, slope, base, length = 28.57, .3761, 5, 5.2085
  nec = geometry(freq, slope, base, length)

  del_theta = 3
  del_phi = 6
  n_theta = 30
  n_phi = 60

  assert 90 % n_theta == 0 and 90 == del_theta * n_theta
  assert 360 % n_phi == 0 and 360 == del_phi * n_phi


  handle_nec(nec_rp_card(nec, 0, n_theta, n_phi+1, 0, 5, 0, 0, 0, 0, del_theta, del_phi, 0, 0))

  thetas = np.linspace(0,90-del_theta,n_theta)
  phis = np.linspace(0,360,n_phi+1)

  rings = []

  for theta_index, theta in enumerate(thetas):
    ring = [nec_gain(nec, 0, theta_index, phi_index) for phi_index, phi in enumerate(phis)]
    rings.append(ring)
             
  max_gain = nec_gain_max(nec, 0)
  min_gain = nec_gain_min(nec, 0)

  nec_delete(nec)

  elevation = [ring[0] for ring in rings]

  fig, axes = plt.subplots(ncols=2, subplot_kw={'projection': 'polar'})

#  ax = fig.add_subplot()

  X = np.cos(np.deg2rad(phis))
  Y = np.sin(np.deg2rad(phis))

  #R = 10**(np.array(rings[-5])/10)


  axes[0].set_aspect(1)

  for i in range(len(rings)):
    R = np.maximum(np.array(rings[i]) - min_gain, 0)
    #ax.plot(R*X, R*Y)

  R = max_gain-min_gain
  #ax.plot(R*X, R*Y)


  for theta, ring in list(zip(thetas, rings))[-11:-8]:
    print(90-theta, np.max(ring))
    axes[0].plot(np.deg2rad(phis),ring,marker='',label=f"{(90-theta):.0f}")

  axes[0].legend(loc="lower left")

  axes[1].set_aspect(1)
  axes[1].plot(np.deg2rad(90-thetas),elevation,marker='')

  plt.show()

def pattern3d():
  freq, slope, base, length = 28.57, .3761, 5, 5.2085
  nec = geometry(freq, slope, base, length)
  handle_nec(nec_rp_card(nec, 0, 15, 61, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0))

  thetas = np.linspace(0,90,15)
  phis = np.linspace(0,360,61)

  rhos = []

  for phi_index, phi in enumerate(phis):
    rho = [nec_gain(nec, 0, theta_index, phi_index) for theta_index, theta in enumerate(thetas)]
    rhos.append(rho)
             
  nec_delete(nec)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')


  Theta, Phi = np.meshgrid(np.deg2rad(thetas),np.deg2rad(phis))
  Rho = 10**(np.array(rhos)/10)

  X = Rho * np.sin(Theta)*np.cos(Phi)
  Y = Rho * np.sin(Theta)*np.sin(Phi)
  Z = Rho * np.cos(Theta)

  ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
  ax.set_aspect('equal')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()



def impedance(freq, slope, base, length):
  nec = geometry(freq, slope, base, length)
  handle_nec(nec_xq_card(nec, 0)) # Execute simulation
  index = 0
  z = complex(nec_impedance_real(nec,index), nec_impedance_imag(nec,index))
  nec_delete(nec)
  return z

def objective(independent_variables, freq, base):
    (length,slope) = independent_variables
    z = impedance(freq, slope, base, length)

    z0 = 200
    reflection_coefficient = (z - z0) / (z + z0)
    rho = abs(reflection_coefficient)
    swr = (1+rho)/(1-rho)
    rho_db = np.log10(rho)*10.0

    print("Impedance at freq = %0.3f, slope=%0.4f, base=%0.4f, length=%0.4f : (%.3f,%+.3fj) Ohms rho=%.4f swr=%.4f, rho_db=%.3f" % (freq, slope, base, length, z.real, z.imag, rho, swr, rho_db))
    return rho_db

def sweep_freq():
  freq, slope, base, length = 28.57, .3761, 7, 5.2085

  xs = np.linspace(28.000,29.000,101)

  nec = geometry(freq, slope, base, length)

  handle_nec(nec_fr_card(nec, 0, 101, 28.000, 1/100))
  handle_nec(nec_xq_card(nec, 0)) # Execute simulation
  
  zs = [complex(nec_impedance_real(nec,index), nec_impedance_imag(nec,index)) for index in range(101)]

  nec_delete(nec)

  zs = np.array(zs)

  z0 = 200

  reflection_coefficient = (zs - z0) / (zs + z0)
  rho = np.abs(reflection_coefficient)
  swr = (1+rho)/(1-rho)

  rho_db = np.log10(rho)*10.0

  fig, ax0 = plt.subplots()
  color = 'tab:red'
  ax0.set_xlabel('freq')
  ax0.set_ylabel('rho_db', color=color)
  ax0.plot(xs, rho_db, color=color)
  ax0.tick_params(axis='y', labelcolor=color)

  color = 'tab:blue'
  ax1 = ax0.twinx()
  ax1.set_ylabel('swr', color=color)
  ax1.plot(xs, swr, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.show()


def sweep_length():
  freq, slope, base, length = 28.57, .3761, 7, 5.2085

  xs = np.linspace(10,12,21)
  zs = [impedance(freq, slope, base, length) for length in xs]
  y0s = [z.real for z in zs]
  y1s = [z.imag for z in zs]
  
  fig, ax0 = plt.subplots()
  color = 'tab:red'
  ax0.set_xlabel('length')
  ax0.set_ylabel('z real', color=color)
  ax0.plot(xs, y0s, color=color)
  ax0.tick_params(axis='y', labelcolor=color)

  color = 'tab:blue'
  ax1 = ax0.twinx()
  ax1.set_ylabel('z imag', color=color)
  ax1.plot(xs, y1s, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.show()





if __name__ == '__main__':

  pattern()
  #pattern3d()
  exit()

  sweep_freq()
  exit()
  #sweep_length()

  freq, slope, base, length = 28.57, .3761, 7, 5.2085

  #'Nelder-Mead'
  #'Powell'
  result = minimize(objective, x0=(length, slope), method='Nelder-Mead', bounds=((4,6),(.3,1)), args=(freq, base), options={'xtol': 0.01})
  print(result)
  length, slope = result.x

  print(objective((length, slope), freq, base))


