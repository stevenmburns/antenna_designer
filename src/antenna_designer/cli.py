from . import AntennaBuilder
from icecream import ic
from . import sweep, sweep_gain, pattern, pattern3d, compare_patterns, optimize

import argparse
from importlib import import_module
from types import ModuleType

def resolve_class(s):
    lst = s.split('.')
    
    """
    Try in order:
    local with explicit Builder
    local with implicit Builder
    library with explicit Builder
    library with implicit Builder
    """

    ic(s, lst)

    def try_to_resolve(builder_name, module_name):
        ic(builder_name, module_name)
        try:
            module = import_module(module_name)
            try:
                res = getattr(module, builder_name)
                ic(res)
                if isinstance(res, ModuleType):
                    return None
                else:
                    return res
            except AttributeError as ex:
                ic(ex)
                return None
        except ModuleNotFoundError as ex:
            ic(ex)
            return None

    if len(lst) > 1:
        if (res := try_to_resolve(lst[-1], '.'.join(lst[:-1]))) is not None:
            return res

    if (res := try_to_resolve('Builder', '.'.join(lst))) is not None:
        return res

    lst = ['antenna_designer', 'designs'] + lst

    if len(lst) > 1:
        if (res := try_to_resolve(lst[-1], '.'.join(lst[:-1]))) is not None:
            return res

    if (res := try_to_resolve('Builder', '.'.join(lst))) is not None:
        return res


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

    def get_builder(nm):
        return resolve_class(nm)

    def get_builders(nms):
        return (get_builder(nm) for nm in nms)


    if args.command == 'draw':
        builder = get_builder(args.builder)
        AntennaBuilder.draw(builder().build_wires(), fn=args.fn)
    elif args.command == 'sweep':
        builder = get_builder(args.builder)
        if args.gain:
            sweep_gain(builder(), args.param, rng=args.range, npoints=args.npoints, center=args.center, fraction=args.fraction, fn=args.fn)
        else:
            sweep(builder(), args.param, rng=args.range, npoints=args.npoints, center=args.center, fraction=args.fraction, fn=args.fn)
    elif args.command == 'pattern':
        builder = get_builder(args.builder)
        if args.wireframe:
            pattern3d(builder(), fn=args.fn)
        else:
            pattern(builder(), fn=args.fn)
    elif args.command == 'compare_patterns':
        builders = get_builders(args.builders)
        compare_patterns((builder() for builder in builders), fn=args.fn)
    elif args.command == 'optimize':
        builder = get_builder(args.builder)
        opt_builder = optimize(builder(), args.params, z0=args.z0, opt_gain=args.opt_gain, resonance=args.resonance)
        print(opt_builder)
        compare_patterns((builder(), opt_builder), fn=args.fn)
