from antenna_designer import AntennaBuilder
import math
from types import MappingProxyType
# from icecream import ic

class Builder(AntennaBuilder):
  default_params = MappingProxyType({
    'freq': 28.57,
    'base': 7,
    'length_20': 10.2551,
    'length_17': 8.2461,
    'length_15': 6.9880,
    'length_12': 5.9681,
    'length_10': 5.2691,
    'freq_20': 14.300,
    'freq_17': 18.1575,
    'freq_15': 21.383,
    'freq_12': 24.97,
    'freq_10': 28.47,
    'slope': 0.5
  })

  def build_wires(self):
    eps = 0.01

    radius = .12
    t0 = radius * math.sqrt(2)

    n = 5

    lst = [(math.cos(math.pi*i/180),math.sin(math.pi*i/180)) for i in range(360//(2*n), 360, 360//n)]
        
    def build_path(lst, ns, ex):
      return ((a,b,ns,ex) for a,b in zip(lst[:-1], lst[1:]))
    def ry(p):
      return  p[0], -p[1], p[2]

    Zc = 1/math.sqrt(1+self.slope**2)
    Zs = self.slope*Zc


    S = (0, eps, 0) 
    T = ry(S)


    C = (S[0], S[1]+t0*Zc, S[2]-t0*Zs)



    A = [(C[0]+radius*x,
          C[1]+radius*y*Zs,
          C[2]+radius*y*Zc
    ) for (x, y) in lst]

    def dist(p0, p1):
      return math.sqrt(sum((x0-x1)**2 for x0, x1 in zip(p0, p1)))
    
    print(f"t0: {t0} dist: {dist(S, C)}")
    print(f"t0: {t0} dists from C: {[dist(C, a) for a in A]}")
    print(f"radius: {radius} dists from S: {[dist(S, a) for a in A]}")

    lengths = [self.length_10, self.length_12, self.length_15, self.length_17, self.length_20]

    ls = [(q/2 - dist(S, a)) for (q,a) in zip(lengths, A)]

    B = [(AA[0], AA[1]+q*Zc, AA[2]-q*Zs) for q, AA in zip(ls, A)]

    Ay = [ry(p) for p in A]
    By = [ry(p) for p in B]

    
    for i in range(n):
      wire_length = dist(S, A[i]) + dist(A[i], B[i])

      print( f"{i} length {wire_length} {lengths[i]/2} {(wire_length-lengths[i]/2)/lengths[i]}")


    n_seg0 = 21
    n_seg1 = 1
      
    tups = []
    for i in range(n):
      tups.extend(build_path([S,A[i],B[i]], n_seg0, None))
      tups.extend(build_path([T,Ay[i],By[i]], n_seg0, None))
    tups.append((T, S, n_seg1, 1+0j))

    new_tups = []
    for (xoff, yoff, zoff) in [(0, 0, self.base)]:
      new_tups.extend([((x0+xoff, y0+yoff, z0+zoff), (x1+xoff, y1+yoff, z1+zoff), ns, ev) for ((x0, y0, z0), (x1, y1, z1), ns, ev) in tups])

    return new_tups

if __name__ == "__main__":

  def tofeet_inches(m):
    f, i = divmod(m/0.0254, 12)

    ii, frac16 = divmod(i*16, 16)

    frac16 = int(frac16+0.5)

    g = math.gcd(frac16, 16)
    factor = 1/1.05

    return f"{m*100:.1f} ({m*100*factor+1:.1f}) ({489.3+90-(m*100*factor+1):.1f}) cm {f:.0f} ft {i:.3f} in ({ii:.0f} {frac16//g}/{16//g} in)"

  params = Builder.default_params

  bands = [20, 17, 15, 12, 10]
  lengths = [f'length_{b}' for b in bands]

  for b in bands:
    length = f"length_{b}"
    print(f"Quarter wave element on {b}m: {tofeet_inches(params[length]/2)}")

