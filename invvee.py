from antenna import *

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

def get_invvee_data():
  return { 'freq': 28.57, 'base': 7, 'length': 5.084, 'slope': 0.604}

def test_invvee_sweep_freq():
  sweep_freq(InvVeeBuilder(**get_invvee_data()), fn='invvee_sweep_freq.pdf')

def test_invvee_sweep_length():
  sweep(InvVeeBuilder(**get_single_bowtie_data()), 'length', (4,6), fn='invvee_sweep_length.pdf')

def test_invvee_sweep_slope():
  sweep(InvVeeBuilder(**get_single_bowtie_data()), 'slope', (4,6), fn='invvee_sweep_slope.pdf')


def test_invvee_optimize():

  gold_params = get_invvee_data()

  params = optimize(InvVeeBuilder(**gold_params), ['length','slope'], z0=50)

  for k, v in gold_params.items():
    assert math.fabs(params[k]-v) < 0.01
