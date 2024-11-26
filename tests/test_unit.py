from antenna_designer.designs.dipole import Builder

from antenna_designer.sweep import resolve_range

def test_unit_params():
  dp = Builder({'freq':1, 'base':7, 'length':10})
  assert dp.freq == 1
  assert dp.base == 7
  assert dp.length == 10

  dp.params['freq'] = 2
  assert dp.freq == 2
  assert dp.params['freq'] == 2

def test_resolve_range():
  # test all eight cases of potential None arguments

  def check(res, gold):
    return all(abs(r-g)<0.01 for r,g in zip(res, gold))

  check(resolve_range(default_value=100, rng=None, center=None, fraction=None), (80, 125))
  check(resolve_range(default_value=100, rng=None, center=30, fraction=None), (24, 37.5))
  check(resolve_range(default_value=100, rng=None, center=None, fraction=1.5), (66.667, 150))
  check(resolve_range(default_value=100, rng=None, center=30, fraction=1.5), (20, 45))
  
  check(resolve_range(default_value=100, rng=(7,11), center=None, fraction=None), (7,11))
  check(resolve_range(default_value=100, rng=(7,11), center=30, fraction=None), (7,11))
  check(resolve_range(default_value=100, rng=(7,11), center=None, fraction=1.5), (7,11))
  check(resolve_range(default_value=100, rng=(7,11), center=30, fraction=1.5), (7,11))

  
