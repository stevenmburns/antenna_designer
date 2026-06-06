from functools import partial

from . import AntennaBuilder
from . import sweep, sweep_gain, sweep_patterns, pattern, pattern3d, compare_patterns, optimize
from .engines import PyNECEngine, PysimEngine

from pysim import TriangularPySim, SinusoidalPySim, BSplinePySim

from icecream import ic

import argparse
from importlib import import_module
from types import ModuleType

ENGINE_CLASSES = {
    'pynec': PyNECEngine,
    'pysim': PysimEngine,
}

PYSIM_BASES = {
    'triangular': TriangularPySim,
    'sinusoidal': SinusoidalPySim,
    'bspline': BSplinePySim,
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


def list_variants(cls):
    """Return all variant names for a Builder class. A variant is any class
    attribute whose name ends in '_params' and is a Mapping; the variant name
    is the attribute name with '_params' stripped."""
    from collections.abc import Mapping
    out = []
    for nm in dir(cls):
        if not nm.endswith('_params'):
            continue
        if not isinstance(getattr(cls, nm), Mapping):
            continue
        out.append(nm[:-len('_params')])
    return sorted(out)


def get_builder(nm):
    """Resolve a builder spec into a zero-arg factory.

    Spec is "name" or "name:variant". A variant binds the named '<variant>_params'
    class attribute as the builder's params; absent or ':default' uses default_params.
    """
    name, _, variant = nm.partition(':')
    cls = resolve_class(name)
    if cls is None:
        return None
    if not variant or variant == 'default':
        return cls
    attr = f'{variant}_params'
    params = getattr(cls, attr, None)
    if params is None:
        available = ', '.join(list_variants(cls)) or '(none)'
        raise ValueError(
            f"builder {name!r} has no variant {variant!r}; available: {available}"
        )
    return partial(cls, params=params)


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


def broadcast_pairs(builders, engines):
    """Numpy-style 1D broadcast of two sequences into a list of pairs.

    Equal lengths zip pairwise; a length-1 sequence broadcasts against the
    other. Any other length mismatch raises.
    """
    nb, ne = len(builders), len(engines)
    if nb == ne:
        return list(zip(builders, engines))
    if nb == 1:
        return [(builders[0], e) for e in engines]
    if ne == 1:
        return [(b, engines[0]) for b in builders]
    raise argparse.ArgumentTypeError(
        f"cannot broadcast {nb} builders against {ne} engines; "
        "lengths must match or one side must be 1"
    )


def parse_engine_spec(spec):
    """Parse an engine spec into (engine_name, kwargs_to_bind).

    Forms: "pynec", "pysim", "pysim:triangular|sinusoidal|bspline".
    """
    name, _, basis = spec.partition(':')
    if name not in ENGINE_CLASSES:
        raise argparse.ArgumentTypeError(
            f"unknown engine {name!r}; available: {', '.join(sorted(ENGINE_CLASSES))}"
        )
    if not basis:
        return name, {}
    if name != 'pysim':
        raise argparse.ArgumentTypeError(
            f"engine {name!r} does not accept a basis suffix (got {basis!r})"
        )
    if basis not in PYSIM_BASES:
        raise argparse.ArgumentTypeError(
            f"unknown pysim basis {basis!r}; available: {', '.join(sorted(PYSIM_BASES))}"
        )
    return name, {'solver': PYSIM_BASES[basis]}


def make_engine_factory(engine_spec, ground_spec):
    name, kwargs = parse_engine_spec(engine_spec)
    cls = ENGINE_CLASSES[name]
    # PyNECEngine's default ground IS finite; pysim's default is free.
    # When the user passes --ground explicitly we always honour it;
    # when they don't, we use whatever the engine's own default is.
    if ground_spec is not _GROUND_UNSET:
        kwargs['ground'] = ground_spec
    if not kwargs:
        return cls
    return partial(cls, **kwargs)


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

    def add_engine_args(p, plural=False):
        if plural:
            p.add_argument('--engines', type=str, nargs='+', default=['pynec'],
                           help='One or more simulation backends. Each spec is '
                                '"pynec" or "pysim[:triangular|sinusoidal|bspline]". '
                                'Cross-products with --builders.')
        else:
            p.add_argument('--engine', type=str, default='pynec',
                           help='Simulation backend: pynec | pysim | '
                                'pysim:triangular | pysim:sinusoidal | pysim:bspline '
                                '(default: pynec).')
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
    add_engine_args(p, plural=True)
    add_pattern_common(p)
    def f(args):
        ground = args.ground if args.ground is _GROUND_UNSET else parse_ground(args.ground)
        pairs = broadcast_pairs(args.builders, args.engines)
        multi_engine = len(set(args.engines)) > 1
        multi_builder = len(set(args.builders)) > 1
        instances = []
        labels = []
        for bname, espec in pairs:
            eng = make_engine_factory(espec, ground)
            instances.append(eng(get_builder(bname)()))
            if multi_engine and multi_builder:
                labels.append(f'{bname}/{espec}')
            elif multi_engine:
                labels.append(espec)
            else:
                labels.append(bname)
        compare_patterns(instances, elevation_angle=args.elevation_angle, fn=args.fn,
                         builder_names=labels, azimuth_f=args.azimuth_f, azimuth_r=args.azimuth_r)
    p.set_defaults(func=f)

    args = parser.parse_args(args=arguments)
    args.func(args)
