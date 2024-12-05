from antenna_designer import pysim
import cProfile
import pstats

with cProfile.Profile() as pr:
    for i in range(1):
        pysim.PySim(nsegs=4001).vectorized_compute_impedance()
    
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
    ps.dump_stats('foo')
    ps.print_stats()
