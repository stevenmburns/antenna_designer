from antenna_designer import pysim

import pprofile
profiler = pprofile.Profile()
with profiler:
    for i in range(1):
        pysim.PySim(nsegs=401).augmented_compute_impedance(ntrap=16)

# You can also write the result to the console:
profiler.print_stats()
profiler.dump_stats("profiler_stats.txt")
