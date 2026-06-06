from functools import partial

from . import AntennaBuilder
from . import sweep, sweep_gain, sweep_patterns, pattern, pattern3d, compare_patterns, optimize
from .engines import PyNECEngine, PysimEngine

from icecream import ic

import argparse
from importlib import import_module
from types import ModuleType

ENGINE_CLASSES = {
    'pynec': PyNECEngine,
    'pysim': PysimEngine,
}


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
                return None if isinstance(res, ModuleType) else res
            except AttributeError:
                return None
        except ModuleNotFoundError:
            return None

    def try_to_resolve_list(lst):
        if len(lst) > 1:
            if (res := try_to_resolve(lst[-1], '.'.join(lst[:-1]))) is not None:
                return res

        return try_to_resolve('Builder', '.'.join(lst))

    if (res := try_to_resolve_list(lst)) is not None:
        return res

    return try_to_resolve_list(['antenna_designer', 'designs'] + lst)


def get_builder(nm):
    return resolve_class(nm)

def get_builders(nms):
    return (get_builder(nm) for nm in nms)


def parse_ground(s):
    """--ground argument:
        free                      -> None
        pec                       -> 'pec'
        finite                    -> default ('finite', 10.0, 0.002)
        finite:<eps_r>,<sigma>    -> ('finite', eps_r, sigma)
    """
    if s is None or s == 'free':
        return None
    if s == 'pec':
        return 'pec'
    if s == 'finite':
        return ('finite', 10.0, 0.002)
    if s.startswith('finite:'):
        try:
            eps_r, sigma = (float(x) for x in s[len('finite:'):].split(','))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"bad --ground spec {s!r}: {e}") from e
        return ('finite', eps_r, sigma)
    raise argparse.ArgumentTypeError(f"unrecognised --ground: {s!r}")


def make_engine_factory(engine_name, ground_spec):
    cls = ENGINE_CLASSES[engine_name]
    # PyNECEngine's default ground IS finite; pysim's default is free.
    # When the user passes --ground explicitly we always honour it;
    # when they don't, we use whatever the engine's own default is.
    if ground_spec is _GROUND_UNSET:
        return cls
    return partial(cls, ground=ground_spec)


_GROUND_UNSET = object()


def cli(arguments=None):

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='command')

    def add_common(p, use_builders=False):
        p.add_argument('--fn', type=str, default=None, help='Plot goes to the file, or displayed on screen if None.')
        if use_builders:
            p.add_argument('--builders', type=str, nargs='+', default=['dipole', 'invvee'], help='Use this list of antenna builders.')
        else:
            p.add_argument('--builder', type=str, default='dipole', help='Use this antenna builder.')

    def add_engine_args(p):
        p.add_argument('--engine', choices=sorted(ENGINE_CLASSES), default='pynec',
                       help='Simulation backend (default: pynec).')
        p.add_argument('--ground', default=_GROUND_UNSET,
                       help='Ground model: free | pec | finite | finite:<eps_r>,<sigma> '
                            '(default: engine-specific — finite for pynec, free for pysim).')


    def add_pattern_common(p):
        p.add_argument('--elevation_angle', default=15, type=float, help='Elevation angle for azimuth plot.')
        p.add_argument('--azimuth_f', default=0, type=int, help='Azimuth angle (front) for the elevation plot.')
        p.add_argument('--azimuth_r', default=180, type=int, help='Azimuth angle (rear) for the elevation plot.')


    def engine_factory_from_args(args):
        ground = args.ground if args.ground is _GROUND_UNSET else parse_ground(args.ground)
        return make_engine_factory(args.engine, ground)


    p = subparsers.add_parser('draw', help='Draw antenna')
    add_common(p)
    def f(args):
        builder = get_builder(args.builder)
        AntennaBuilder.draw(builder().build_wires(), fn=args.fn)
    p.set_defaults(func=f)

    p = subparsers.add_parser('sweep', help='Sweep antenna')
    add_common(p)
    add_engine_args(p)
    add_pattern_common(p)
    p.add_argument('--param', type=str, default='freq', help='Variable to sweep.')
    p.add_argument('--range', nargs=2, default=None, type=float, help='Range for sweep.')
    p.add_argument('--center', default=None, type=float, help='Center if range not given.')
    p.add_argument('--fraction', default=None, type=float, help='Fraction around center for range.')
    p.add_argument('--npoints', default=21, type=int, help='Points in the range.')
    p.add_argument('--gain', default=False, action='store_true', help='Plot gain instead of impedance.')
    p.add_argument('--use_smithchart', default=False, action='store_true', help='Plot impedance using a smithchart.')
    p.add_argument('--z0', default=50, type=float, help='Reference impedance.')
    p.add_argument('--markers', default=[], nargs='+', type=float, help='Add markers at these values.')

    p.add_argument('--patterns', default=False, action='store_true', help='Compare patterns generated for each swept value.')

    def f(args):
        builder = get_builder(args.builder)
        engine = engine_factory_from_args(args)
        if args.patterns:
            sweep_patterns(builder(), args.param, rng=args.range, npoints=args.npoints, center=args.center, fraction=args.fraction, fn=args.fn, elevation_angle=args.elevation_angle, azimuth_f=args.azimuth_f, azimuth_r=args.azimuth_r, engine=engine)
        elif args.gain:
            sweep_gain(builder(), args.param, rng=args.range, npoints=args.npoints, center=args.center, fraction=args.fraction, fn=args.fn, engine=engine)
        else:
            sweep(builder(), args.param, rng=args.range, npoints=args.npoints, center=args.center, fraction=args.fraction, use_smithchart=args.use_smithchart, fn=args.fn, z0=args.z0, markers=args.markers, engine=engine)
    p.set_defaults(func=f)

    p = subparsers.add_parser('optimize', help='Optimize antenna')
    add_common(p)
    add_engine_args(p)
    p.add_argument('--params', nargs='+', default=None, type=str, help='Use these optimization params.')
    p.add_argument('--z0', default=50, type=float, help='Use this reference impedance.')
    p.add_argument('--resonance', default=False, action='store_true', help='Optimize to resonance instead of matching an impedance.')
    p.add_argument('--opt_gain', default=False, action='store_true', help='Also try to optimize gain.')
    def f(args):
        builder = get_builder(args.builder)
        engine = engine_factory_from_args(args)
        opt_builder = optimize(builder(), args.params, z0=args.z0, opt_gain=args.opt_gain, resonance=args.resonance, engine=engine)
        print(opt_builder)
        compare_patterns([engine(builder()), engine(opt_builder)], fn=args.fn)
    p.set_defaults(func=f)

    p = subparsers.add_parser('pattern', help='Display far field of antenna')
    add_common(p)
    add_engine_args(p)
    p.add_argument('--wireframe', default=False, action='store_true', help='Draw wireframe.')
    p.add_argument('--elevation_angle', default=15, type=float, help='Elevation angle for azimuth plot.')
    def f(args):
        builder = get_builder(args.builder)
        engine = engine_factory_from_args(args)
        if args.wireframe:
            pattern3d(engine(builder()), fn=args.fn)
        else:
            pattern(engine(builder()), elevation_angle=args.elevation_angle, fn=args.fn)
    p.set_defaults(func=f)

    p = subparsers.add_parser('compare_patterns', help='Display far field of multiple antennas')
    add_common(p, use_builders=True)
    add_engine_args(p)
    add_pattern_common(p)
    def f(args):
        builders = get_builders(args.builders)
        engine = engine_factory_from_args(args)
        compare_patterns([engine(builder()) for builder in builders], elevation_angle=args.elevation_angle, fn=args.fn, builder_names=args.builders, azimuth_f=args.azimuth_f, azimuth_r=args.azimuth_r)
    p.set_defaults(func=f)

    args = parser.parse_args(args=arguments)
    args.func(args)
