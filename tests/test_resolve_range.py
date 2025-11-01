
from antenna_designer import resolve_range, gen_xs
import numpy as np


def test_range():
    assert (.8, 1.25) == resolve_range(None, (.8, 1.25), None, None)
    assert (.8, 1.25) == resolve_range(None, None, 1, None)
    assert (.8, 1.25) == resolve_range(None, None, 1, 1.25)

def test_gen_xs():    
    assert np.allclose(gen_xs(None, (.8, 1.25), None, None, 2), np.array( (0.8, 1.25)))
    assert np.allclose(gen_xs(None, (.8, 1.25), None, None, 1), np.array( (0.8, )))
    assert np.allclose(gen_xs(None, (.8, .8), None, None, 1), np.array( (0.8, )))
