from . import AntennaBuilder
from icecream import ic
from . import sweep, sweep_gain, pattern, pattern3d, compare_patterns, optimize
from . import designs
from .designs import * #noqa F401, F403

import argparse

def cli(arguments=None):
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='command')

    parser_draw = subparsers.add_parser('draw', help='Draw antenna')
    parser_draw.add_argument('--fn', type=str, default=None, help='Plot goes to the file, or displayed on screen if None.')
    parser_draw.add_argument('--builder', type=str, default='dipole', help='Use this antenna builder.')


    parser_sweep = subparsers.add_parser('sweep', help='Sweep antenna')
    parser_sweep.add_argument('--fn', type=str, default=None, help='Plot goes to the file, or displayed on screen if None.')
    parser_sweep.add_argument('--builder', type=str, default='dipole', help='Use this antenna builder.')
    parser_sweep.add_argument('--param', type=str, default='freq', help='Use this sweep parameter.')
    parser_sweep.add_argument('--range', nargs=2, default=None, type=float, help='Use this sweep range.')
    parser_sweep.add_argument('--center', default=None, type=float, help='Use this to construct the sweep range.')
    parser_sweep.add_argument('--fraction', default=None, type=float, help='Use this to construct the sweep range.')
    parser_sweep.add_argument('--npoints', default=21, type=int, help='Use this as the number of points in the sweep.')
    parser_sweep.add_argument('--gain', default=False, action='store_true', help='Plot gain instead of impedance.')

    parser_optimize = subparsers.add_parser('optimize', help='Optimize antenna')
    parser_optimize.add_argument('--fn', type=str, default=None, help='Plot goes to the file, or displayed on screen if None.')
    parser_optimize.add_argument('--builder', type=str, default='dipole', help='Use this antenna builder.')
    parser_optimize.add_argument('--params', nargs='+', default=None, type=str, help='Use these optimization params.')
    parser_optimize.add_argument('--z0', default=50, type=float, help='Use this reference impedance.')
    parser_optimize.add_argument('--resonance', default=False, action='store_true', help='Optimize to resonance instead of matching an impedance.')
    parser_optimize.add_argument('--opt_gain', default=False, action='store_true', help='Also try to optimize gain.')

    parser_pattern = subparsers.add_parser('pattern', help='Display far field of antenna')
    parser_pattern.add_argument('--fn', type=str, default=None, help='Plot goes to the file, or displayed on screen if None.')
    parser_pattern.add_argument('--builder', type=str, default='dipole', help='Use this antenna builder.')
    parser_pattern.add_argument('--wireframe', default=False, action='store_true', help='Draw wireframe.')

    p = subparsers.add_parser('compare_patterns', help='Display far field of multiple antennas')
    p.add_argument('--fn', type=str, default=None, help='Plot goes to the file, or displayed on screen if None.')
    p.add_argument('--builders', type=str, nargs='+', default=['dipole', 'invvee'], help='Use this list of antenna builders.')

    args = parser.parse_args(args=arguments)

    ic(args)

    if args.command == 'draw':
        module = getattr(designs, args.builder)
        b = module.Builder()
        AntennaBuilder.draw(b.build_wires(), fn=args.fn)
    elif args.command == 'sweep':
        module = getattr(designs, args.builder)
        if args.gain:
            sweep_gain(module.Builder(), args.sweep_param, rng=args.sweep_range, npoints=args.sweep_npoints, center=args.sweep_center, fraction=args.sweep_fraction, fn=args.fn)
        else:
            sweep(module.Builder(), args.param, rng=args.range, npoints=args.npoints, center=args.center, fraction=args.fraction, fn=args.fn)
    elif args.command == 'pattern':
        module = getattr(designs, args.builder)
        if args.wireframe:
            pattern3d(module.Builder(), fn=args.fn)
        else:
            pattern(module.Builder(), fn=args.fn)
    elif args.command == 'compare_patterns':
        modules = (getattr(designs, builder) for builder in args.builders)
        builders = (module.Builder() for module in modules)
        compare_patterns(builders, fn=args.fn)
    elif args.command == 'optimize':
        module = getattr(designs, args.builder)
        builder = optimize(module.Builder(), args.params, z0=args.z0, opt_gain=args.opt_gain, resonance=args.resonance)
        print(builder)
        compare_patterns((module.Builder(), builder), fn=args.fn)
