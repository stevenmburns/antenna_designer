from . import sweep, sweep_gain, sweep_freq, pattern, pattern3d, compare_patterns
from . import designs
from .designs import * #noqa F401, F403

import argparse

parser = argparse.ArgumentParser()

commands = [
    'draw', 'sweep', 'sweep_gain', 'sweep_freq', 'pattern', 'pattern3d', 'compare_patterns'
]


parser.add_argument('command', choices=commands, help='Command to run.')

parser.add_argument('--builder', type=str, default='dipole', help='Use this antenna builder.')
parser.add_argument('--builders', type=str, nargs='+', default=['dipole', 'invvee'], help='Use this list of antenna builders.')

parser.add_argument('--sweep_param', type=str, default='freq', help='Use this sweep parameter.')

parser.add_argument('--sweep_range', nargs='+', default=None, type=float, help='Use this sweep range.')

parser.add_argument('--sweep_npoints', default=21, type=int, help='Use this as the number of points in the sweep.')

args = parser.parse_args()

if args.command == 'compare_patterns':
    modules = (getattr(designs, builder) for builder in args.builders)
else:
    module = getattr(designs, args.builder)

if args.command == 'draw':
    b = module.Builder()
    b.draw(b.build_wires())
elif args.command == 'sweep':
    sweep(module.Builder(), args.sweep_param, rng=args.sweep_range, npoints=args.sweep_npoints)
elif args.command == 'sweep_freq':
    sweep_freq(module.Builder(), rng=args.sweep_range, npoints=args.sweep_npoints)
elif args.command == 'sweep_gain':
    sweep_gain(module.Builder(), args.sweep_param, rng=args.sweep_range, npoints=args.sweep_npoints)
elif args.command == 'pattern':
    pattern(module.Builder())
elif args.command == 'pattern3d':
    pattern3d(module.Builder())
elif args.command == 'compare_patterns':
    builders = (module.Builder() for module in modules)
    compare_patterns(builders)
