from dipole import DipoleBuilder

def test_unit_params():
  dp = DipoleBuilder({'freq':1, 'base':7, 'length':10})
  assert dp.freq == 1
  assert dp.base == 7
  assert dp.length == 10

  dp.params['freq'] = 2
  assert dp.freq == 2
