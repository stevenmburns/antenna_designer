import PyNEC as nec

class Antenna:
  def __init__(self, antenna_builder):
    self.builder = antenna_builder
    self.tups = self.builder.build_wires()
    self.tls = self.builder.build_tls()
    self.excitation_pairs = None
    self.geometry()

  def __del__(self):
    del self.c

  def geometry(self):

    conductivity = 5.8e7 # Copper
    ground_conductivity = 0.002
    ground_dielectric = 10

    self.c = nec.nec_context()

    geo = self.c.get_geometry()

    self.excitation_pairs = []
    for idx, (p0, p1, n_seg, ev) in enumerate(self.tups, start=1):
      geo.wire(idx, n_seg, p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], 0.0005, 1.0, 1.0)
      if ev is not None:
        self.excitation_pairs.append((idx, (n_seg+1)//2, ev))

    self.c.geometry_complete(0)

    for (idx1, seg1, idx2, seg2, impedance, length) in self.tls:
      self.c.tl_card(idx1, seg1, idx2, seg2, impedance, length, 0, 0, 0, 0)

    self.c.ld_card(5, 0, 0, 0, conductivity, 0.0, 0.0)
    self.c.gn_card(0, 0, ground_dielectric, ground_conductivity, 0, 0, 0, 0)

    for tag, sub_index, voltage in self.excitation_pairs:
      self.c.ex_card(0, tag, sub_index, 0, voltage.real, voltage.imag, 0, 0, 0, 0)

  def set_freq_and_execute(self):
    self.c.fr_card(0, 1, self.builder.freq, 0)
    self.c.xq_card(0) # Execute simulation

  def impedance(self, freq_index=0, sum_currents=False, sweep=False):
    if not sweep:
      self.set_freq_and_execute()

    sc = self.c.get_structure_currents(freq_index)

    indices = []
    for tag, tag_index, voltage in self.excitation_pairs:
      matches = [(i, t) for (i, t) in enumerate(sc.get_current_segment_tag()) if t == tag]
      index = matches[tag_index-1][0]
      indices.append((index, voltage))

    currents = sc.get_current()

    zs = [voltage/currents[idx] for idx, voltage in indices]

    if sum_currents:
      zs = [1/sum(1/z for z in zs)]

    return zs
