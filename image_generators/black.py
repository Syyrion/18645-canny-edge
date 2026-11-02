#!/usr/bin/env python3

import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Generate a solid black image.")
parser.add_argument("width", type=int, help="Image width in pixels")
parser.add_argument("height", type=int, help="Image height in pixels")
parser.add_argument("-o", "--output", default="black_image.png", help="Output filename")
args = parser.parse_args()

if args.width <= 0 or args.height <= 0:
    parser.error("width and height must be positive integers")

Image.new("RGB", (args.width, args.height), color="black").save(args.output)
