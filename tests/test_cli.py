
import antenna_designer as ant

o = ' --fn /dev/null'
#o = ''

def test_cli():

  ant.cli(f'draw{o}'.split())
  ant.cli(f'draw --builder moxon{o}'.split())
  ant.cli(f'sweep --param tipspacer_factor --builder moxon{o}'.split())
  ant.cli(f'optimize --params length slope --builder invvee{o}'.split())
  ant.cli(f'optimize --params length slope --z0 60 --builder invvee{o}'.split())
  ant.cli(f'compare_patterns{o}'.split())
  ant.cli(f'compare_patterns --builders invvee moxon{o}'.split())
  ant.cli(f'compare_patterns --builders invvee hexbeam{o}'.split())
