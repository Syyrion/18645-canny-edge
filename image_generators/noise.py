#!/usr/bin/env python3
"""Generate a white-noise image with the requested width and height."""

import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("width", type=int)
parser.add_argument("height", type=int)
parser.add_argument("-o", "--output", default="white_noise.png")
parser.add_argument("--seed", type=int, help="Optional seed for reproducible output")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
pixels = rng.integers(0, 256, size=(args.height, args.width), dtype=np.uint8)
Image.fromarray(pixels, mode="L").save(args.output)
