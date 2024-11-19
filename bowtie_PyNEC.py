from PyNEC import *
import math
import numpy as np

from scipy.optimize import minimize_scalar, minimize

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d


class Antenna:
  def __init__(self):
    self.excitation_pairs = None
    self.geometry()

  def __del__(self):
    del self.c

  def draw(self):

    pairs = [(p0,p1) for p0, p1, _, _ in self.tups]

    lc = LineCollection(pairs, colors=(1, 0, 0, 1), linewidths=1)

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale(axis='x')
    ax.set_aspect('equal')
    ax.margins(0.1)
    plt.show()



  def geometry(self):

    conductivity = 5.8e7 # Copper
    ground_conductivity = 0.002
    ground_dielectric = 10

    self.c = nec_context()

    geo = self.c.get_geometry()

    self.excitation_pairs = []
    for idx, (p0, p1, n_seg, ex) in enumerate(self.tups, start=1):
      geo.wire(idx, n_seg, 0, p0[0], p0[1], 0, p1[0], p1[1], 0.002, 1.0, 1.0)
      if ex:
        self.excitation_pairs.append((idx, (n_seg+1)//2))

    self.c.geometry_complete(0)

    self.c.ld_card(5, 0, 0, 0, conductivity, 0.0, 0.0)
    self.c.gn_card(0, 0, ground_dielectric, ground_conductivity, 0, 0, 0, 0)

    for tag, sub_index in self.excitation_pairs:
      self.c.ex_card(0, tag, sub_index, 0, 1.0, 0, 0, 0, 0, 0)

  def set_freq_and_execute(self):
    self.c.fr_card(0, 1, self.freq, 0)
    self.c.xq_card(0) # Execute simulation

  def impedance(self, freq_index=0, sum_currents=False, sweep=False):
    if not sweep:
      self.set_freq_and_execute()

    sc = self.c.get_structure_currents(freq_index)

    indices = []
    for tag, tag_index in self.excitation_pairs:
      matches = [(i, t) for (i, t) in enumerate(sc.get_current_segment_tag()) if t == tag]
      index = matches[tag_index-1][0]
      indices.append(index)

    currents = sc.get_current()

    if sum_currents:
      zs = [1/sum(currents[idx] for idx in indices)]
    else:
      zs = [1/currents[idx] for idx in indices]

    return zs


class Bowtie(Antenna):
  
  def __init__(self, freq, slope, base, length):
    self.freq, self.slope, self.base, self.length = freq, slope, base, length
    self.tups = self.build_wires()
    super().__init__()

  def __del__(self):
    del self.c

  def build_wires(self):
    eps = 0.05

    # diag = sqrt(x^2 + (x*slope)^2) = x*sqrt(1+slope^2)
    # length/2 = diag + x*slope = x*(slope + sqrt(1+slope^2))

    x = 0.5*self.length/(self.slope + math.sqrt(1+self.slope**2))
    z = self.slope*x

    n_seg0 = 21
    n_seg1 = 3

    tups = []
    tups.extend([((-x,    0),   (-x,   z),    n_seg0, False)])
    tups.extend([((-x,    z),   (-eps, eps),  n_seg0, False)])
    tups.extend([((-eps,  eps), ( eps, eps),  n_seg1, False)])
    tups.extend([(( eps,  eps), ( x,   z),    n_seg0, False)])
    tups.extend([(( x,    z),   ( x,   0),    n_seg0, False)])
    tups.extend([((-x,    0),   (-x,   -z),   n_seg0, False)])
    tups.extend([((-x,   -z),   (-eps, -eps), n_seg0, False)])
    tups.extend([(( eps, -eps), ( x,   -z),   n_seg0, False)])
    tups.extend([(( x,   -z),   ( x,    0),   n_seg0, False)])
    tups.extend([((-eps, -eps), ( eps, -eps), n_seg1, True)])

    new_tups = []
    for (xoff, yoff) in [(-4, self.base+2), (-4, self.base-2), (4, self.base+2), (4, self.base-2)]:
      new_tups.extend([((x0+xoff, y0+yoff), (x1+xoff, y1+yoff), ns, ex) for ((x0, y0), (x1, y1), ns, ex) in tups])

    return new_tups

class BowtieSingle(Antenna):
  
  def __init__(self, freq, slope, base, length):
    self.freq, self.slope, self.base, self.length = freq, slope, base, length
    self.tups = self.build_wires()
    super().__init__()

  def __del__(self):
    del self.c

  def build_wires(self):
    eps = 0.05

    # diag = sqrt(x^2 + (x*slope)^2) = x*sqrt(1+slope^2)
    # length/2 = diag + x*slope = x*(slope + sqrt(1+slope^2))

    x = 0.5*self.length/(self.slope + math.sqrt(1+self.slope**2))
    z = self.slope*x

    n_seg0 = 21
    n_seg1 = 3

    tups = []
    tups.extend([((-x,    0),   (-x,   z),    n_seg0, False)])
    tups.extend([((-x,    z),   (-eps, eps),  n_seg0, False)])
    tups.extend([((-eps,  eps), ( eps, eps),  n_seg1, False)])
    tups.extend([(( eps,  eps), ( x,   z),    n_seg0, False)])
    tups.extend([(( x,    z),   ( x,   0),    n_seg0, False)])
    tups.extend([((-x,    0),   (-x,   -z),   n_seg0, False)])
    tups.extend([((-x,   -z),   (-eps, -eps), n_seg0, False)])
    tups.extend([(( eps, -eps), ( x,   -z),   n_seg0, False)])
    tups.extend([(( x,   -z),   ( x,    0),   n_seg0, False)])
    tups.extend([((-eps, -eps), ( eps, -eps), n_seg1, True)])

    new_tups = []
    for (xoff, yoff) in [(0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff), (x1+xoff, y1+yoff), ns, ex) for ((x0, y0), (x1, y1), ns, ex) in tups])

    return new_tups

class Dipole(Antenna):
  
  def __init__(self, freq, slope, base, length):
    self.freq, self.slope, self.base, self.length = freq, slope, base, length
    self.tups = self.build_wires()
    super().__init__()

  def __del__(self):
    del self.c

  def build_wires(self):
    eps = 0.05

    x = 0.5*self.length

    n_seg0 = 21
    n_seg1 = 3

    tups = []
    tups.extend([((-x,   0), (-eps, 0), n_seg0, False)])
    tups.extend([(( eps, 0), ( x,   0),    n_seg0, False)])
    tups.extend([((-eps, 0),  ( eps, 0), n_seg1, True)])

    new_tups = []
    for (xoff, yoff) in [(0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff), (x1+xoff, y1+yoff), ns, ex) for ((x0, y0), (x1, y1), ns, ex) in tups])

    return new_tups



def pattern(antenna, antenna_params, fn=None):
  freq, slope, base, length = antenna_params

  bt = antenna(freq, slope, base, length)
  bt.set_freq_and_execute()

  del_theta = 3
  del_phi = 6
  n_theta = 30
  n_phi = 60

  assert 90 % n_theta == 0 and 90 == del_theta * n_theta
  assert 360 % n_phi == 0 and 360 == del_phi * n_phi


  bt.c.rp_card(0, n_theta, n_phi+1, 0, 5, 0, 0, 0, 0, del_theta, del_phi, 0, 0)

  thetas = np.linspace(0,90-del_theta,n_theta)
  phis = np.linspace(0,360,n_phi+1)

  rings = []

  for theta_index, theta in enumerate(thetas):
    ring = [bt.c.get_gain(0, theta_index, phi_index) for phi_index, phi in enumerate(phis)]
    rings.append(ring)
             
  max_gain = bt.c.get_gain_max(0)
  min_gain = bt.c.get_gain_min(0)

  del bt

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


  for theta, ring in list(zip(thetas, rings))[-7:-1]:
    print(90-theta, np.max(ring))
    axes[0].plot(np.deg2rad(phis),ring,marker='',label=f"{(90-theta):.0f}")

  axes[0].legend(loc="lower left")

  axes[1].set_aspect(1)
  axes[1].plot(np.deg2rad(90-thetas),elevation,marker='')

  if fn is not None:
    plt.savefig(fn)
  else:
    plt.show()


def pattern3d(antenna, antenna_params, fn=None):
  freq, slope, base, length = antenna_params

  bt = antenna(freq, slope, base, length)
  bt.set_freq_and_execute()

  del_theta = 3
  del_phi = 6
  n_theta = 30
  n_phi = 60

  assert 90 % n_theta == 0 and 90 == del_theta * n_theta
  assert 360 % n_phi == 0 and 360 == del_phi * n_phi

  bt.c.rp_card(0, n_theta, n_phi+1, 0, 5, 0, 0, 0, 0, del_theta, del_phi, 0, 0)

  thetas = np.linspace(0,90-del_theta,n_theta)
  phis = np.linspace(0,360,n_phi+1)

  rhos = [[bt.c.get_gain(0, theta_index, phi_index) for theta_index, _ in enumerate(thetas)] for phi_index, _ in enumerate(phis)]
             
  del bt

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

  if fn is not None:
    plt.savefig(fn)
  else:
    plt.show()



def sweep_freq(antenna, antenna_params, fn=None):

  min_freq = 28.0
  max_freq = 29.0
  n_freq = 20
  del_freq = (max_freq- min_freq)/n_freq

  xs = np.linspace(min_freq, max_freq, n_freq+1)
  
  freq, slope, base, length = antenna_params
  bt = antenna(freq, slope, base, length)

  bt.c.fr_card(0, n_freq+1, min_freq, del_freq)
  bt.c.xq_card(0) # Execute simulation
  
  zs = np.array([bt.impedance(freq_index,sweep=True) for freq_index in range(len(xs))])

  del bt

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

  if fn is not None:
    plt.savefig(fn)
  else:
    plt.show()


def sweep_length(antenna, antenna_params, fn=None):
  freq, slope, base, length = antenna_params

  xs = np.linspace(5.5,6.0,21)
  zs = np.array([antenna(freq, slope, base, length).impedance() for length in xs])
  
  fig, ax0 = plt.subplots()
  color = 'tab:red'
  ax0.set_xlabel('length')
  ax0.set_ylabel('z real', color=color)
  ax0.tick_params(axis='y', labelcolor=color)
  for i in range(zs.shape[1]):
    ax0.plot(xs, np.real(zs)[:,i], color=color)


  color = 'tab:blue'
  ax1 = ax0.twinx()
  ax1.set_ylabel('z imag', color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  for i in range(zs.shape[1]):
    ax1.plot(xs, np.imag(zs)[:,i], color=color)

  fig.tight_layout()

  if fn is not None:
    plt.savefig(fn)
  else:
    plt.show()

def sweep_slope(antenna, antenna_params, fn=None):
  freq, slope, base, length = antenna_params
  xs = np.linspace(.4,.8,21)
  zs = np.array([antenna(freq, slope, base, length).impedance() for slope in xs])
  
  fig, ax0 = plt.subplots()
  color = 'tab:red'
  ax0.set_xlabel('slope')
  ax0.set_ylabel('z real', color=color)
  ax0.tick_params(axis='y', labelcolor=color)
  for i in range(zs.shape[1]):
    ax0.plot(xs, np.real(zs)[:,i], color=color)


  color = 'tab:blue'
  ax1 = ax0.twinx()
  ax1.set_ylabel('z imag', color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  for i in range(zs.shape[1]):
    ax1.plot(xs, np.imag(zs)[:,i], color=color)

  fig.tight_layout()

  if fn is not None:
    plt.savefig(fn)
  else:
    plt.show()

def optimize(antenna, antenna_params):
  freq, slope, base, length = antenna_params

  def objective(independent_variables, freq, base):
      (length,slope) = independent_variables
      zs = antenna(freq, slope, base, length).impedance()

      z0 = 200
      for z in zs:
        reflection_coefficient = (z - z0) / (z + z0)
        rho = abs(reflection_coefficient)
        swr = (1+rho)/(1-rho)
        rho_db = np.log10(rho)*10.0

        print("Impedance at freq = %0.3f, slope=%0.4f, base=%0.4f, length=%0.4f : (%.3f,%+.3fj) Ohms rho=%.4f swr=%.4f, rho_db=%.3f" % (freq, slope, base, length, z.real, z.imag, rho, swr, rho_db))

      return sum([abs(z - z0) for z in zs])

  #'Nelder-Mead'
  #'Powell', options={'xtol': 0.01}
  result = minimize(objective, x0=(length, slope), method='Nelder-Mead', bounds=((4,6),(.3,1)), args=(freq, base))
  print(result)
  length, slope = result.x

  print(objective((length, slope), freq, base))

  return freq, slope, base, length


def optimize1(antenna, antenna_params):
  freq, slope, base, length = antenna_params

  def objective(independent_variables):
      (length,) = independent_variables
      zs = antenna(freq, slope, base, length).impedance()

      z0 = 50
      for z in zs:
        reflection_coefficient = (z - z0) / (z + z0)
        rho = abs(reflection_coefficient)
        swr = (1+rho)/(1-rho)
        rho_db = np.log10(rho)*10.0

        print("Impedance at freq = %0.3f, slope=%0.4f, base=%0.4f, length=%0.4f : (%.3f,%+.3fj) Ohms rho=%.4f swr=%.4f, rho_db=%.3f" % (freq, slope, base, length, z.real, z.imag, rho, swr, rho_db))

      #return sum([abs(z - z0) for z in zs])
      return sum([abs(z.imag) for z in zs])

  #'Nelder-Mead'
  #'Powell', options={'xtol': 0.01}
  result = minimize(objective, x0=(length,), method='Nelder-Mead', bounds=((4,6),))
  print(result)
  length, = result.x

  print(objective((length,)))

  return freq, slope, base, length


def get_data():
  freq, slope, base, length = 28.57, .6608, 7, 5.7913
  return freq, slope, base, length


def test_pattern():
  pattern(Bowtie, get_data(), fn='pattern.pdf')

def test_pattern3d():
  pattern3d(Bowtie, get_data(), fn='pattern3d.pdf')
  
def test_sweep_freq():
  sweep_freq(Bowtie, get_data(), fn='sweep_freq.pdf')

def test_sweep_slope():
  sweep_slope(Bowtie, get_data(), fn='sweep_slope.pdf')

def test_sweep_length():
  sweep_length(Bowtie, get_data(), fn='sweep_length.pdf')

def test_optimize():
  save_params = optimize(Bowtie, get_data())
  assert all(math.fabs(x-y) < 0.01 for x,y in zip(get_data(), save_params))

def test_single_sweep_freq():
  sweep_freq(BowtieSingle, get_data(), fn='single_sweep_freq.pdf')

def test_dipole_sweep_freq():
  sweep_freq(Dipole, get_data(), fn='dipole_sweep_freq.pdf')

def test_dipole_pattern():
  pattern(Dipole, get_data(), fn='dipole_pattern.pdf')

def test_dipole_pattern3d():
  pattern3d(Dipole, get_data(), fn='dipole_pattern3d.pdf')

def test_dipole_optimize():
  gold_params =  28.57, .6608, 7, 5.0325
  save_params = optimize1(Dipole, get_data())

  assert all(math.fabs(x-y) < 0.01 for x,y in zip(gold_params, save_params))

if __name__ == '__main__':
  pass


