from antenna import *

def get_bowtie_data():
  return { 'freq': 28.57, 'slope_top': .658, 'slope_bot': .512, 'base': 7, 'length_top': 5.771, 'length_bot': 5.68}

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
    for yoff in [-4, 4]:
      zoff = self.base+2
      new_tups.extend([((0, y0+yoff, z0+zoff), (0, y1+yoff, z1+zoff), ns, ex) for ((y0, z0), (y1, z1), ns, ex) in tups_top])
      zoff = self.base-2
      new_tups.extend([((0, y0+yoff, z0+zoff), (0, y1+yoff, z1+zoff), ns, ex) for ((y0, z0), (y1, z1), ns, ex) in tups_bot])

    return new_tups

def get_single_bowtie_data():
  return { 'freq': 28.57, 'slope': .363, 'base': 7, 'length': 5.185}

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
    for (yoff, zoff) in [(0, self.base)]:
      new_tups.extend([((0, y0+yoff, z0+zoff), (0, y1+yoff, z1+zoff), ns, ex) for ((y0, z0), (y1, z1), ns, ex) in tups])

    return new_tups

