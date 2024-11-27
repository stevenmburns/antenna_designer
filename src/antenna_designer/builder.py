from .core import save_or_show

import matplotlib.pyplot as plt
#from matplotlib.collections import LineCollection
#from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class AntennaBuilder:
  def __init__(self, params=None):
    # write directly to __dict__ because otherwise __setattr__ goes into infinite loop
    self.__dict__['_params'] = dict(
      self.__class__.default_params if params is None else params
    )

    "Check that params key's are legal"
    assert all(k in self.__class__.default_params for k in self._params.keys())

  def __getattr__(self, nm):
    return self._params[nm]

  def __setattr__(self, nm, v):
    self._params[nm] = v

  def __str__(self):
    res = []
    for k, v in self._params.items():
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
    self.__dict__['element_builder'] = element_builder
    super().__init__(params)

  def build_wires(self):
    elem_params = self.element_builder.default_params
    elem_params_keys = set(elem_params.keys())

    changed_keys = set()
    for k,v in self._params.items():
      if k not in elem_params_keys:
        if k.endswith('_top') or k.endswith('_bot'):
          elem_key = k[:-4]
          assert elem_key in elem_params_keys
          changed_keys.add(elem_key)

    def build_element_wires(suffix):
      local_element_params = dict(elem_params)
      for k,v in self._params.items():    
        if k in elem_params_keys and k not in changed_keys:
          local_element_params[k] = v

      for k in changed_keys:
        local_element_params[k] = self._params[k + suffix]

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
    self.__dict__['element_builder'] = element_builder
    super().__init__(params)

  def build_wires(self):
    elem_params = self.element_builder.default_params
    elem_params_keys = set(elem_params.keys())

    suffixes = ['_itop', '_ibot', '_otop', '_obot']

    changed_keys = set()
    for k,v in self._params.items():
      if k not in elem_params_keys:
        if any(k.endswith(suffix) for suffix in suffixes):
          elem_key = k[:-5]
          assert elem_key in elem_params_keys
          changed_keys.add(elem_key)

    def build_element_wires(suffix):
      local_element_params = dict(elem_params)
      for k,v in self._params.items():    
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

