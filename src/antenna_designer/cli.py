from . import AntennaBuilder
from icecream import ic
from . import sweep, sweep_gain, pattern, pattern3d, compare_patterns, optimize
from . import designs
from .designs import * #noqa F401, F403

import argparse

def cli(arguments=None):
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='command')

    def add_common(p, use_builders=False):
        p.add_argument('--fn', type=str, default=None, help='Plot goes to the file, or displayed on screen if None.')
        if use_builders:
            p.add_argument('--builders', type=str, nargs='+', default=['dipole', 'invvee'], help='Use this list of antenna builders.')
        else:
            p.add_argument('--builder', type=str, default='dipole', help='Use this antenna builder.')


    p = subparsers.add_parser('draw', help='Draw antenna')
    add_common(p)

    p = subparsers.add_parser('sweep', help='Sweep antenna')
    add_common(p)
    p.add_argument('--param', type=str, default='freq', help='Use this sweep parameter.')
    p.add_argument('--range', nargs=2, default=None, type=float, help='Use this sweep range.')
    p.add_argument('--center', default=None, type=float, help='Use this to construct the sweep range.')
    p.add_argument('--fraction', default=None, type=float, help='Use this to construct the sweep range.')
    p.add_argument('--npoints', default=21, type=int, help='Use this as the number of points in the sweep.')
    p.add_argument('--gain', default=False, action='store_true', help='Plot gain instead of impedance.')

    p = subparsers.add_parser('optimize', help='Optimize antenna')
    add_common(p)
    p.add_argument('--params', nargs='+', default=None, type=str, help='Use these optimization params.')
    p.add_argument('--z0', default=50, type=float, help='Use this reference impedance.')
    p.add_argument('--resonance', default=False, action='store_true', help='Optimize to resonance instead of matching an impedance.')
    p.add_argument('--opt_gain', default=False, action='store_true', help='Also try to optimize gain.')

    p = subparsers.add_parser('pattern', help='Display far field of antenna')
    add_common(p)
    p.add_argument('--wireframe', default=False, action='store_true', help='Draw wireframe.')

    p = subparsers.add_parser('compare_patterns', help='Display far field of multiple antennas')
    add_common(p, use_builders=True)

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
