try:
    from .engines.pynec import PyNECEngine
except ImportError:
    PyNECEngine = None

Antenna = PyNECEngine
