
import antenna_designer as ant
from antenna_designer.designs.dipole import Builder

import sys

def test_cli():
  ant.cli('draw --fn /dev/null'.split())
  ant.cli('draw --builder moxon --fn /dev/null'.split())
  ant.cli('sweep --param tipspacer_factor --builder moxon --fn /dev/null'.split())
  ant.cli('optimize --params length slope --builder invvee --fn /dev/null'.split())
  ant.cli('optimize --params length slope --z0 60 --builder invvee --fn /dev/null'.split())
  ant.cli('compare_patterns --fn /dev/null'.split())
