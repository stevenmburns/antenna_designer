from antenna_designer.designs.dipole import Builder

from types import MappingProxyType

from antenna_designer.sweep import resolve_range

def test_unit_params():
  dp = Builder({'freq':1, 'base':7, 'length':10})
  assert dp.freq == 1
  assert dp.base == 7
  assert dp.length == 10

  dp._params['freq'] = 2
  assert dp.freq == 2
  assert dp._params['freq'] == 2

  dp.freq = 3
  assert dp.freq == 3
  assert dp._params['freq'] == 3

  dp.z0 = 50
  assert dp.z0 == 50
  assert dp._params['z0'] == 50



def test_dict_update_options():
  p = {'a': 0, 'b': 1}

  q = dict(p, **{'b':2})
  assert p['a'] == 0 and p['b'] == 1
  assert q['a'] == 0 and q['b'] == 2

  r = dict(p, b=2)
  assert p['a'] == 0 and p['b'] == 1
  assert r['a'] == 0 and r['b'] == 2

  s = dict(p)
  s['b'] = 2
  assert p['a'] == 0 and p['b'] == 1
  assert s['a'] == 0 and s['b'] == 2

  p = MappingProxyType({'a': 0, 'b': 1})

  q = dict(p, **{'b':2})
  assert p['a'] == 0 and p['b'] == 1
  assert q['a'] == 0 and q['b'] == 2

  r = dict(p, b=2)
  assert p['a'] == 0 and p['b'] == 1
  assert r['a'] == 0 and r['b'] == 2

  s = dict(p)
  s['b'] = 2
  assert p['a'] == 0 and p['b'] == 1
  assert s['a'] == 0 and s['b'] == 2
  


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

  
