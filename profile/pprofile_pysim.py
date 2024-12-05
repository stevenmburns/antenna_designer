from antenna_designer import pysim

import pprofile
profiler = pprofile.Profile()
with profiler:
    for i in range(1):
        pysim.PySim(nsegs=4001).vectorized_compute_impedance()

# You can also write the result to the console:
profiler.print_stats()
profiler.dump_stats("profiler_stats.txt")
