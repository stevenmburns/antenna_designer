from PyNEC import *
import math
import numpy as np

from scipy.optimize import minimize_scalar, minimize

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d

class AntennaBuilder:
  def __init__(self, freq):
    self.params = {}
    self.params['freq'] = freq

  def __getattr__(self, nm):
    return self.params[nm]

  def __str__(self):
    res = []
    for k, v in self.params.items():
      res.append(f"{k} = {v:0.3f}")
    return ', '.join(res)


class Antenna:
  def __init__(self, antenna_builder):
    self.params = antenna_builder
    self.tups = self.params.build_wires()
    self.excitation_pairs = None
    self.geometry()

  def __del__(self):
    del self.c

  def __getattr__(self, nm):
    return self.params.__getattr__(nm)

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


class BowtieBuilder(AntennaBuilder):
  def __init__(self, freq, slope_top, slope_bot, base, length_top, length_bot):
    super().__init__(freq)
    self.params['slope_top'] = slope_top
    self.params['slope_bot'] = slope_bot
    self.params['base'] = base
    self.params['length_top'] = length_top
    self.params['length_bot'] = length_bot

  def build_wires(self):
    eps = 0.05

    n_seg0 = 21
    n_seg1 = 3

    def element(length, slope):
      # diag = sqrt(x^2 + (x*slope)^2) = x*sqrt(1+slope^2)
      # length/2 = diag + x*slope = x*(slope + sqrt(1+slope^2))

      x = 0.5*length/(slope + math.sqrt(1+slope**2))
      z = slope*x

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
      return tups

    tups_top = element(self.length_top, self.slope_top)
    tups_bot = element(self.length_bot, self.slope_bot)

    new_tups = []
    for xoff in [-4, 4]:
      yoff = self.base+2
      new_tups.extend([((x0+xoff, y0+yoff), (x1+xoff, y1+yoff), ns, ex) for ((x0, y0), (x1, y1), ns, ex) in tups_top])
      yoff = self.base-2
      new_tups.extend([((x0+xoff, y0+yoff), (x1+xoff, y1+yoff), ns, ex) for ((x0, y0), (x1, y1), ns, ex) in tups_bot])

    return new_tups

class BowtieSingleBuilder(AntennaBuilder):
  def __init__(self, freq, slope, base, length):
    super().__init__(freq)
    self.params['slope'] = slope
    self.params['base'] = base
    self.params['length'] = length

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


class DipoleBuilder(AntennaBuilder):
  def __init__(self, freq, base, length):
    super().__init__(freq)
    self.params['base'] = base
    self.params['length'] = length

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


class InvVeeBuilder(AntennaBuilder):
  def __init__(self, freq, slope, base, length):
    super().__init__(freq)
    self.params['slope'] = slope
    self.params['base'] = base
    self.params['length'] = length

  def build_wires(self):
    eps = 0.05

    # (0.5*self.length)^2 == x^2+z^2
    # z = self.slope*x
    # (0.5*self.length)^2 == x^2*(1+self.slope^2)
    # 0.5*self.length == x*sqrt(1+self.slope^2)

    x = 0.5*self.length/math.sqrt(1+self.slope**2)
    z = self.slope*x

    n_seg0 = 21
    n_seg1 = 3

    tups = []
    tups.extend([((-x,  -z), (-eps, 0), n_seg0, False)])
    tups.extend([(( eps, 0), ( x,  -z), n_seg0, False)])
    tups.extend([((-eps, 0), ( eps, 0), n_seg1, True)])

    new_tups = []
    for (xoff, yoff) in [(0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff), (x1+xoff, y1+yoff), ns, ex) for ((x0, y0), (x1, y1), ns, ex) in tups])

    return new_tups


def pattern(antenna_builder, fn=None):
  bt = Antenna(antenna_builder)
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


def pattern3d(antenna_builder, fn=None):
  bt = Antenna(antenna_builder)
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



def sweep_freq(antenna_builder, fn=None):

  min_freq = 28.0
  max_freq = 29.0
  n_freq = 20
  del_freq = (max_freq- min_freq)/n_freq

  xs = np.linspace(min_freq, max_freq, n_freq+1)
  
  bt = Antenna(antenna_builder)

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


def sweep(antenna_builder, nm, rng, npoints=21, fn=None):

  xs = np.linspace(rng[0],rng[1],npoints)

  zs = []
  for x in xs:
    antenna_builder.params[nm] = x
    zs.append(Antenna(antenna_builder).impedance())
  zs = np.array(zs)
  
  fig, ax0 = plt.subplots()
  color = 'tab:red'
  ax0.set_xlabel(nm)
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

def sweep_length(antenna_builder, fn=None):

  xs = np.linspace(5.5,6.0,21)

  zs = []
  for length in xs:
    antenna_builder.params['length'] = length
    zs.append(Antenna(antenna_builder).impedance())
  zs = np.array(zs)
  
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

def sweep_slope(antenna_builder, fn=None):
  xs = np.linspace(.4,.8,21)
  zs = []
  for slope in xs:
    antenna_builder.params['slope'] = slope
    zs.append(Antenna(antenna_builder).impedance())
  zs = np.array(zs)

  
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

def optimize(antenna_builder, independent_variable_names, z0=50, resonance=False):

  def objective(independent_variables):

      for v, nm in zip(independent_variables, independent_variable_names):
        antenna_builder.params[nm] = v

      zs = Antenna(antenna_builder).impedance()

      for z in zs:
        reflection_coefficient = (z - z0) / (z + z0)
        rho = abs(reflection_coefficient)
        swr = (1+rho)/(1-rho)
        rho_db = np.log10(rho)*10.0


        print("Impedance at %s: (%.3f,%+.3fj) Ohms rho=%.4f swr=%.4f, rho_db=%.3f" % (str(antenna_builder), z.real, z.imag, rho, swr, rho_db))


      if resonance:
        return sum([abs(z.imag) for z in zs])
      else:
        return sum([abs(z - z0) for z in zs])

  #'Nelder-Mead'
  #'Powell', options={'maxiter':100, 'disp': True, 'xtol': 0.0001}

  x0 = tuple(antenna_builder.params[nm] for nm in independent_variable_names)

  bounds = tuple((x*.6, x*1.67) for x in x0)

  result = minimize(objective, x0=x0, method='Nelder-Mead', tol=0.001, bounds=bounds)

  print(result)

  for x, nm in zip(result.x, independent_variable_names):
    antenna_builder.params[nm] = x

  print(objective(result.x))

  return antenna_builder.params


def get_dipole_data():
  return { 'freq': 28.57, 'base': 7, 'length': 5.032}

def get_invvee_data():
  return { 'freq': 28.57, 'base': 7, 'length': 5.084, 'slope': 0.604}

def get_bowtie_data():
  return { 'freq': 28.57, 'slope_top': .658, 'slope_bot': .512, 'base': 7, 'length_top': 5.771, 'length_bot': 5.68}

def get_single_bowtie_data():
  return { 'freq': 28.57, 'slope': .363, 'base': 7, 'length': 5.185}


def test_bowtie_pattern():
  pattern(BowtieBuilder(**get_bowtie_data()), fn='pattern.pdf')

def test_bowtie_pattern3d():
  pattern3d(BowtieBuilder(**get_bowtie_data()), fn='pattern3d.pdf')
  
def test_bowtie_sweep_freq():
  sweep_freq(BowtieBuilder(**get_bowtie_data()), fn='sweep_freq.pdf')

def test_bowtie_sweep_freq2():
  sweep(BowtieBuilder(**get_bowtie_data()), 'freq', (28,29), fn='sweep_freq.pdf')

def test_bowtie_sweep_slope_top():
  sweep(BowtieBuilder(**get_bowtie_data()), 'slope_top', (.2,1), fn='bowtie_sweep_slope_top.pdf')

def test_bowtie_sweep_slope_bot():
  sweep(BowtieBuilder(**get_bowtie_data()), 'slope_bot', (.2,1), fn='bowtie_sweep_slope_bot.pdf')

def test_bowtie_sweep_length_top():
  sweep(BowtieBuilder(**get_bowtie_data()), 'length_top', (4,6), fn='bowtie_sweep_length_top.pdf')

def test_bowtie_sweep_length_bot():
  sweep(BowtieBuilder(**get_bowtie_data()), 'length_bot', (4,6), fn='bowtie_sweep_length_bot.pdf')

def test_bowtie_optimize():
  gold_params = get_bowtie_data()

  params = optimize(BowtieBuilder(**gold_params), ['length_top', 'slope_top', 'length_bot', 'slope_bot'], z0=200)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01


def test_single_sweep_freq():
  sweep_freq(BowtieBuilder(**get_single_bowtie_data()), fn='single_sweep_freq.pdf')

def test_single_pattern():
  pattern(BowtieBuilder(**get_single_bowtie_data()), fn='single_pattern.pdf')

def test_single_pattern3d():
  pattern3d(BowtieSingleBuilder(**get_single_bowtie_data()), fn='single_pattern3d.pdf')

def test_single_sweep_slope():
  sweep_slope(BowtieSingleBuilder(**get_single_bowtie_data()), fn='sweep_slope.pdf')

def test_single_sweep_length():
  sweep_length(BowtieSingleBuilder(**get_single_bowtie_data()), fn='sweep_length.pdf')

def test_single_optimize():
  gold_params = get_single_bowtie_data()

  params = optimize(BowtieSingleBuilder(**gold_params), ['length', 'slope'], z0=200)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01


def test_dipole_sweep_freq():
  sweep_freq(DipoleBuilder(**get_dipole_data()), fn='dipole_sweep_freq.pdf')

def test_dipole_pattern():
  pattern(DipoleBuilder(**get_dipole_data()), fn='dipole_pattern.pdf')

def test_dipole_pattern3d():
  pattern3d(DipoleBuilder(**get_dipole_data()), fn='dipole_pattern3d.pdf')

def test_dipole_optimize():
  params = optimize(DipoleBuilder(**get_dipole_data()), ['length'], z0=50, resonance=True)

  for k, v in get_dipole_data().items():
    assert math.fabs(params.params[k]-v) < 0.01

def test_invvee_optimize():
  gold_params = get_invvee_data()

  params = optimize(InvVeeBuilder(**gold_params), ['length','slope'], z0=50)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01



def test_unit_params():
  dp = DipoleBuilder(freq=1, base=7, length=10)
  assert dp.freq == 1
  assert dp.base == 7
  assert dp.length == 10

  dp.params['freq'] = 2
  assert dp.freq == 2


if __name__ == '__main__':
  pass


