from antenna import *
from bowtie import *



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


