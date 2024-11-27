
import antenna_designer as ant

o = ' --fn /dev/null'
#o = ''

def test_cli():

  for design in [
      'moxon', 'vertical', 'invvee', 'invveearray',
      'bowtie', 'bowtiearray', 'bowtiearray2x4',
      'rawdipole', 'freq_based_yagi', 'freq_based_invvee',
      'fandipole'
  ]:
    ant.cli(f'draw --builder {design}{o}'.split())

  ant.cli(f'sweep --param tipspacer_factor --builder moxon{o}'.split())
  ant.cli(f'sweep --gain --param tipspacer_factor --builder moxon{o}'.split())
  ant.cli(f'optimize --params length slope --builder invvee{o}'.split())
  ant.cli(f'optimize --opt_gain --params length slope --resonance --builder invvee{o}'.split())
  ant.cli(f'pattern --builder freq_based_yagi{o}'.split())
  ant.cli(f'pattern --builder freq_based_invvee --wireframe{o}'.split())
  ant.cli(f'compare_patterns{o}'.split())
  ant.cli(f'compare_patterns --builders invvee moxon{o}'.split())
  ant.cli(f'compare_patterns --builders invvee hexbeam{o}'.split())
