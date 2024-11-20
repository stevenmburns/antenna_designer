from antenna import *



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
  sweep_freq(BowtieSingleBuilder(**get_single_bowtie_data()), fn='single_sweep_freq.pdf')

def test_single_pattern():
  pattern(BowtieSingleBuilder(**get_single_bowtie_data()), fn='single_pattern.pdf')

def test_single_pattern3d():
  pattern3d(BowtieSingleBuilder(**get_single_bowtie_data()), fn='single_pattern3d.pdf')

def test_single_sweep_slope():
  sweep(BowtieSingleBuilder(**get_single_bowtie_data()), 'slope', (.2,1), fn='single_sweep_slope.pdf')

def test_single_sweep_length():
  sweep(BowtieSingleBuilder(**get_single_bowtie_data()), 'length', (4,6), fn='sigle_sweep_length.pdf')


def test_single_optimize():
  gold_params = get_single_bowtie_data()

  params = optimize(BowtieSingleBuilder(**gold_params), ['length', 'slope'], z0=200)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01
