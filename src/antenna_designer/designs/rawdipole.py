
class Builder:
  def __init__(self, freq=28.57, length=5, base=7):
      self.freq = freq
      self.length = length
      self.base = base

  def build_wires(self):
    eps = 0.05
    y = 0.5*self.length
    z = self.base

    n_seg0 = 21
    n_seg1 = 3

    return [
        ((0, -y,   z), (0, -eps, z), n_seg0, None),
        ((0, eps,  z), (0, y,    z), n_seg0, None),
        ((0, -eps, z), (0, eps,  z), n_seg1, 1+0j)
    ]
