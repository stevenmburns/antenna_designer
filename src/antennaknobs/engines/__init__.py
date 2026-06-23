try:
    from .pynec import PyNECEngine
except ImportError:
    PyNECEngine = None

from .momwire import MomwireEngine

__all__ = ["PyNECEngine", "MomwireEngine"]
