#
#  Simple vertical monopole antenna simulation using python-necpp
#  pip install necpp
#
from necpp import *
import math
import numpy as np

from scipy.optimize import minimize_scalar, minimize

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def handle_nec(result):
  if (result != 0):
    print(nec_error_message())

def draw(pairs):
  lc = LineCollection(pairs, colors=(1, 0, 0, 1), linewidths=1)

  fig, ax = plt.subplots()
  ax.add_collection(lc)
  ax.autoscale(axis='x')
  ax.set_aspect('equal')
  ax.margins(0.1)
  plt.show()

  exit()


def geometry(freq, slope, base, length):
  
  conductivity = 5.8e7 # Copper
  ground_conductivity = 0.002
  ground_dielectric = 10

  wavelength = 3e8/(1e6*freq)
  n_seg = 50
  eps = 0.01

  # diag = sqrt(x^2 + (x*slope)^2) = x*sqrt(1+slope^2)
  # length/2 = diag + x*slope = x*(slope + sqrt(1+slope^2))

  x = 0.5*length/(slope + math.sqrt(1+slope**2))
  z = slope*x

  pairs = []
  pairs.extend([((-x, base), (-x, base+z)), ((-x, base+z), (-eps, base+eps))])
  pairs.extend([((-x, base), (-x, base-z)), ((-x, base-z), (-eps, base-eps))])
  pairs.extend([(( x, base), ( x, base+z)), (( eps, base+eps), ( x, base+z))])
  pairs.extend([(( x, base), ( x, base-z)), (( eps, base-eps), ( x, base-z))])
  pairs.extend([((-eps, base+eps), ( eps, base+eps))])
  pairs.extend([((-eps, base-eps), ( eps, base-eps))])
  draw(pairs)
  print(len(pairs))

  nec = nec_create()

  for idx, (p0, p1) in zip(range(1,len(pairs)+1), pairs):
    print(idx, (p0, p1))
    handle_nec(nec_wire(nec, idx, n_seg, 0, p0[0], p0[1], 0, p1[0], p1[1], 0.002, 1.0, 1.0))

  handle_nec(nec_geometry_complete(nec, 1))
  handle_nec(nec_ld_card(nec, 5, 0, 0, 0, conductivity, 0.0, 0.0))
  handle_nec(nec_gn_card(nec, 0, 0, ground_dielectric, ground_conductivity, 0, 0, 0, 0))
  handle_nec(nec_fr_card(nec, 0, 1, freq, 0))
  handle_nec(nec_ex_card(nec, 0, len(pairs), len(pairs), 0, 1.0, 0, 0, 0, 0, 0)) 

  return nec

def impedance(freq, slope, base, length):
  nec = geometry(freq, slope, base, length)
  handle_nec(nec_xq_card(nec, 0)) # Execute simulation
  index = 0
  z = complex(nec_impedance_real(nec,index), nec_impedance_imag(nec,index))
  nec_delete(nec)
  return z

def objective(independent_variables, freq, base):
    (length, slope) = independent_variables
    z = impedance(freq, slope, base, length)
    print("Impedance at freq = %0.2f, slope=%0.2f, base=%0.2f, length=%0.2f : (%6.1f,%+6.1fI) Ohms" % (freq, slope, base, length, z.real, z.imag))
    return abs(z - 200)

if __name__ == '__main__':
  freq, slope, base, length = 14.3, .5, 10, 10.5

  objective((length, slope), freq, base)

  exit()

  result = minimize(objective, x0=(length, slope), method='Powell', bounds=((9,11),(0,1)), args=(freq, base), options={'xtol': 0.01})
  print(result)
  length, slope = result.x

  z = impedance(freq = freq, slope=slope, base = base, length = length)
  print("Impedance at freq = %0.2f, slope=%0.2f, base=%0.2f, length=%0.2f : (%6.1f,%+6.1fI) Ohms" % (freq, slope, base, length, z.real, z.imag))





