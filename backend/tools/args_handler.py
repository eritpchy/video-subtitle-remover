import argparse
from enum import Enum

from .constant import InpaintMode

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Command Line Tool"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o", required=False, type=str, default=None,
        help="Output video file path (optional)"
    )
    parser.add_argument(
        "--ymin", type=int, default=None,
        help="Subtitle area ymin (optional)"
    )
    parser.add_argument(
        "--ymax", type=int, default=None,
        help="Subtitle area ymax (optional)"
    )
    parser.add_argument(
        "--xmin", type=int, default=None,
        help="Subtitle area xmin (optional)"
    )
    parser.add_argument(
        "--xmax", type=int, default=None,
        help="Subtitle area xmax (optional)"
    )
    parser.add_argument(
        "--inpaint-mode", type=str, default="sttn-auto",
        choices=[mode.name.lower().replace('_','-') for mode in InpaintMode],
        help="Inpaint mode, default is sttn-auto"
    )
    args = parser.parse_args()
    args.inpaint_mode = InpaintMode[args.inpaint_mode.replace('-','_').upper()]
    return args