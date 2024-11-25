from . import designs
from .designs import * #noqa F401, F403

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--builder', type=str, default='dipole', help='Use this antenna builder.')

args = parser.parse_args()

module = getattr(designs, args.builder)

b = module.Builder()
b.draw(b.build_wires())


