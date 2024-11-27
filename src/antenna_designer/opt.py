from . import Antenna
from .far_field import get_elevation

import numpy as np

#from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

def optimize(antenna_builder, independent_variable_names, *, z0=50, resonance=False, opt_gain=False, bounds=None):

  def objective(independent_variables):

      for v, nm in zip(independent_variables, independent_variable_names):
        setattr(antenna_builder, nm, v)

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

  x0 = tuple(getattr(antenna_builder, nm) for nm in independent_variable_names)
  if bounds is None:
    bounds = tuple((x*.6, x*1.67) for x in x0)

  result = minimize(objective, x0=x0, method='Nelder-Mead', tol=0.001, bounds=bounds)

  print(result)

  for x, nm in zip(result.x, independent_variable_names):
    setattr(antenna_builder, nm, x)

  print(objective(result.x))

  return antenna_builder

