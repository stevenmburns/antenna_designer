import numpy as np
from antenna_designer.pysim import Spline_Integral_Standalone

def test_spline_integral():
    n = np.array([[0,0,0],[0,0.5,0],[0,1,0],[0,1.5,0],[0,1,0]])
    m = n

    Spline_Integral_Standalone(n, m, ntrap=2, wire_radius=0.0005, k=1.0)

def test_spline_impedance():
    n = np.array([[0,0,0],[0,0.5,0],[0,1,0],[0,1.5,0],[0,1,0]])
    m = n

    Spline_Integral_Standalone(n, m, ntrap=2, wire_radius=0.0005, k=1.0)

