import PyNEC as nec
import numpy as np

#from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

import matplotlib.pyplot as plt
#from matplotlib.collections import LineCollection
#from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def save_or_show(plt, fn):
  if fn is not None:
    plt.savefig(fn)
  else:
    plt.show()

  plt.close()


class AntennaBuilder:
  def __init__(self, params=None):
    if params is None:
      self.params = dict(self.__class__.default_params)
    else:
      self.params = dict(params)

    "Check that params key's are legal"
    assert all(k in self.__class__.default_params for k in self.params.keys())

  def __getattr__(self, nm):
    return self.params[nm]

  def __str__(self):
    res = []
    for k, v in self.params.items():
      res.append(f"{k} = {v:0.3f}")
    return ', '.join(res)

  def draw(self, tups, fn=None):

    pairs = [(p0, p1) for p0, p1, _, _ in tups]
    print(pairs)

    lc = Line3DCollection(pairs, colors=(1, 0, 0, 1), linewidths=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(lc)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(4, 10)
    ax.set_aspect('equal')

    save_or_show(plt, fn)

class Array2x2Builder(AntennaBuilder):
  def __init__(self, element_builder, params=None):
    self.element_builder = element_builder
    super().__init__(params)

  def build_wires(self):
    elem_params = self.element_builder.default_params
    elem_params_keys = set(elem_params.keys())

    changed_keys = set()
    for k,v in self.params.items():
      if k not in elem_params_keys:
        if k.endswith('_top') or k.endswith('_bot'):
          elem_key = k[:-4]
          assert elem_key in elem_params_keys
          changed_keys.add(elem_key)

    def build_element_wires(suffix):
      local_element_params = dict(elem_params)
      for k,v in self.params.items():    
        if k in elem_params_keys and k not in changed_keys:
          local_element_params[k] = v

      for k in changed_keys:
        local_element_params[k] = self.params[k + suffix]

      element_builder_local = self.element_builder(local_element_params)

      return element_builder_local.build_wires()

    tups_top = build_element_wires('_top')
    tups_bot = build_element_wires('_bot')

    new_tups = []
    for yoff in (-self.del_y, self.del_y):
      for zoff, tups in ((self.del_z, tups_top), (-self.del_z, tups_bot)):
        new_tups.extend([((x0, y0+yoff, z0+zoff), (x1, y1+yoff, z1+zoff), ns, ex) for ((x0, y0, z0), (x1, y1, z1), ns, ex) in tups])

    return new_tups

class Array2x4Builder(AntennaBuilder):
  def __init__(self, element_builder, params=None):
    self.element_builder = element_builder
    super().__init__(params)

  def build_wires(self):
    elem_params = self.element_builder.default_params
    elem_params_keys = set(elem_params.keys())

    suffixes = ['_itop', '_ibot', '_otop', '_obot']

    changed_keys = set()
    for k,v in self.params.items():
      if k not in elem_params_keys:
        if any(k.endswith(suffix) for suffix in suffixes):
          elem_key = k[:-5]
          assert elem_key in elem_params_keys
          changed_keys.add(elem_key)

    def build_element_wires(suffix):
      local_element_params = dict(elem_params)
      for k,v in self.params.items():    
        if k in elem_params_keys and k not in changed_keys:
          local_element_params[k] = v

      for k in changed_keys:
        local_element_params[k] = self.params[k + suffix]

      element_builder_local = self.element_builder(local_element_params)

      return element_builder_local.build_wires()

    tups_itop = build_element_wires('_itop')
    tups_otop = build_element_wires('_otop')
    tups_ibot = build_element_wires('_ibot')
    tups_obot = build_element_wires('_obot')

    new_tups = []
    for yoff, pairs in ((-3*self.del_y, ((self.del_z, tups_otop), (-self.del_z, tups_obot))),
                        (-1*self.del_y, ((self.del_z, tups_itop), (-self.del_z, tups_ibot))), 
                        ( 1*self.del_y, ((self.del_z, tups_itop), (-self.del_z, tups_ibot))),
                        ( 3*self.del_y, ((self.del_z, tups_otop), (-self.del_z, tups_obot)))
    ):
      for zoff, tups in pairs:
        new_tups.extend([((x0, y0+yoff, z0+zoff), (x1, y1+yoff, z1+zoff), ns, ex) for ((x0, y0, z0), (x1, y1, z1), ns, ex) in tups])

    return new_tups


class Antenna:
  def __init__(self, antenna_builder):
    self.builder = antenna_builder
    self.tups = self.builder.build_wires()
    self.excitation_pairs = None
    self.geometry()

  def __del__(self):
    del self.c

  def __getattr__(self, nm):
    return self.builder.__getattr__(nm)

  def geometry(self):

    conductivity = 5.8e7 # Copper
    ground_conductivity = 0.002
    ground_dielectric = 10

    self.c = nec.nec_context()

    geo = self.c.get_geometry()

    self.excitation_pairs = []
    for idx, (p0, p1, n_seg, ex) in enumerate(self.tups, start=1):
      geo.wire(idx, n_seg, p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], 0.002, 1.0, 1.0)
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

def build_and_get_elevation(antenna_builder):
  a = Antenna(antenna_builder)
  a.set_freq_and_execute()
  return get_elevation(a)

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

def compare_patterns(antenna_builders, elevation_angle=15, fn=None):
  rings_lst = []

  for antenna_builder in antenna_builders:
    rings, max_gain, min_gain, thetas, phis = get_pattern_rings(antenna_builder)
    rings_lst.append(rings)

  fig, axes = plt.subplots(ncols=2, subplot_kw={'projection': 'polar'})

  axes[0].set_rticks([-12, -6, 0, 6, 12])

  for rings in rings_lst:
    for theta, ring in list(zip(thetas, rings)):
      if abs(theta-(90-elevation_angle)) < 0.1:
        axes[0].plot(np.deg2rad(phis),ring,marker='',label=f"{(90-theta):.0f}")

  axes[0].legend(loc="lower left")

  n = len(rings_lst[0][0])
  assert (n-1) % 2 == 0
  elevations = [list(reversed([ring[0] for ring in rings]))+[ring[(n-1)//2] for ring in rings] for rings in rings_lst]
  el_thetas = list(reversed(list(90-thetas))) + list(90+thetas)

  axes[1].set_rticks([-12, -6, 0, 6, 12])

  for elevation in elevations:
    axes[1].plot(np.deg2rad(el_thetas),elevation,marker='')

  save_or_show(plt, fn)

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

def sweep_freq(antenna_builder, *, z0=200, rng=(28, 29), npoints=21, fn=None):

  if rng is None:
    center = antenna_builder.params['freq']
    rng = (center*.8, center*1.25)

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


def sweep_gain(antenna_builder, nm, rng, npoints=21, fn=None):

  if rng is None:
    center = antenna_builder.params[nm]
    rng = (center*.8, center*1.25)

  xs = np.linspace(rng[0],rng[1],npoints)

  gs = []
  for x in xs:
    antenna_builder.params[nm] = x
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

def sweep(antenna_builder, nm, rng=None, npoints=21, fn=None):

  if rng is None:
    center = antenna_builder.params[nm]
    rng = (center*.8, center*1.25)

  xs = np.linspace(rng[0],rng[1],npoints)

  zs = []
  for x in xs:
    print(nm, x)
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

  save_or_show(plt, fn)


def optimize(antenna_builder, independent_variable_names, z0=50, resonance=False, opt_gain=False):

  print(antenna_builder.params)

  def objective(independent_variables):

      for v, nm in zip(independent_variables, independent_variable_names):
        antenna_builder.params[nm] = v

      a = Antenna(antenna_builder)
      zs = a.impedance()
      _, max_gain, _, _, _ = get_elevation(a)      
      del a


      for z in zs:
        reflection_coefficient = (z - z0) / (z + z0)
        rho = abs(reflection_coefficient)
        swr = (1+rho)/(1-rho)
        rho_db = np.log10(rho)*10.0


        if opt_gain:
          print("Impedance at %s: (%.3f,%+.3fj) Ohms rho=%.4f swr=%.4f, rho_db=%.3f max_gain=%.2f" % (str(antenna_builder), z.real, z.imag, rho, swr, rho_db, max_gain))
        else:
          print("Impedance at %s: (%.3f,%+.3fj) Ohms rho=%.4f swr=%.4f, rho_db=%.3f" % (str(antenna_builder), z.real, z.imag, rho, swr, rho_db))

      res = 0
      if resonance:
        res += sum([abs(z.imag) for z in zs])
      else:
        res += sum([abs(z - z0) for z in zs])

      if opt_gain:
        res -= 100*max_gain

      return res

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

