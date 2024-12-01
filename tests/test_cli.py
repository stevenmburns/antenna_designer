
import antenna_designer as ant

o = ' --fn /dev/null'
#o = ''

def test_cli_draw():
  for design in [
      'moxon', 'vertical', 'invvee', 'invveearray',
      'bowtie', 'bowtiearray', 'bowtiearray2x4',
      'rawdipole', 'freq_based.yagi', 'freq_based.invvee',
      'fandipole'
  ]:
    ant.cli(f'draw --builder {design}{o}'.split())

def test_cli_sweep():
  ant.cli(f'sweep --param tipspacer_factor --builder moxon --npoints 2{o}'.split())
  ant.cli(f'sweep --gain --param tipspacer_factor --npoints 2 --builder moxon{o}'.split())

  ant.cli(f'sweep --markers 28.57 --npoints 0{o}'.split())
  ant.cli(f'sweep --markers 28.57 --npoints 0 --builder invveearray{o}'.split())
  ant.cli(f'sweep --markers 28.57 --npoints 2 --builder invveearray{o}'.split())
  ant.cli(f'sweep --npoints 2 --builder invveearray{o}'.split())

  ant.cli(f'sweep --markers 28.57 --npoints 0{o} --use_smithchart --z0=50'.split())
  ant.cli(f'sweep --markers 28.57 --npoints 0 --builder invveearray{o} --use_smithchart --z0=50'.split())
  ant.cli(f'sweep --markers 28.57 --npoints 2 --builder invveearray{o} --use_smithchart --z0=50'.split())
  ant.cli(f'sweep --npoints 2 --builder invveearray{o} --use_smithchart --z0=50'.split())

def test_cli_optimize():
  ant.cli(f'optimize --params length slope --builder invvee{o}'.split())
  ant.cli(f'optimize --opt_gain --params length slope --resonance --builder invvee{o}'.split())

def test_cli_pattern():
  ant.cli(f'pattern --builder freq_based.yagi{o}'.split())
  ant.cli(f'pattern --builder freq_based.invvee --wireframe{o}'.split())

def test_cli_compare_patterns():
  ant.cli(f'compare_patterns{o}'.split())
  ant.cli(f'compare_patterns --builders invvee moxon{o}'.split())
  ant.cli(f'compare_patterns --builders invvee hexbeam{o}'.split())
