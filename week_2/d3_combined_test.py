from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

from src.internship.fit_result import FitResult
from src.internship.spectrum import SpectrumPlotter, SpectralLine, SpectrumProcessor, LINE_CATALOG
from src.internship.utilities import gaussian_dip, sigmoid_dip , voigt_dip
from src.internship.fitters import LineFitter
from src.internship.spectrum import BatchRunner


matplotlib.use("Agg")
from scipy.optimize import curve_fit
from scipy.special import wofz

try:
    from astropy.io import fits
except ImportError as e:
    raise ImportError(
        "astropy is required to read FITS files. Install with: "
        "pip install astropy --break-system-packages"
    ) from e


def lorentzian_dip(x, amp, cen, gamma, offset):
    return offset - amp * (gamma ** 2 / ((x - cen) ** 2 + gamma ** 2))


def _voigt_peak_normalized(x, cen, sigma, gamma):
    sigma = max(sigma, 1e-6)
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z))
    peak = np.real(wofz(1j * gamma / (sigma * np.sqrt(2))))
    peak = peak if peak > 1e-12 else 1e-12
    return profile / peak
















def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically fit recurring spectral lines (occurrence >= 16/19) "
                    "in one or more SDSS FITS spectra."
    )
    parser.add_argument("files", nargs="*", help="Path(s) to FITS file(s)")
    parser.add_argument("--folder", type=str, default=None,
                        help="Folder containing .fits files to process (all files in it)")
    parser.add_argument("--out", type=str, default="line_fit_results.csv",
                        help="Output CSV path (default: line_fit_results.csv)")
    parser.add_argument("--plots-dir", type=str, default=None,
                        help="If given, save a PNG per file (spectrum + fitted line panels) "
                             "into this folder. Omit this flag to skip plotting.")
    return parser.parse_args()


def main():
    args = parse_args()

    filepaths = list(args.files)
    if args.folder:
        filepaths.extend(sorted(glob.glob(os.path.join(args.folder, "*.fits"))))

    if not filepaths:
        print("No FITS files given. Use positional args or --folder /path/to/dir")
        sys.exit(1)

    plotter = SpectrumPlotter(args.plots_dir) if args.plots_dir else None
    processor = SpectrumProcessor(LINE_CATALOG, plotter=plotter)
    runner = BatchRunner(processor=processor)
    df = runner.run(filepaths, out_csv=args.out)

    print("\n=== Summary ===")
    print(f"Files processed : {df['file_name'].nunique()}")
    print(f"Total line fits : {len(df)}")
    print(f"Successful fits : {int(df['success'].sum())}")
    print(f"Results saved to: {args.out}")
    if args.plots_dir:
        print(f"Plots saved to  : {args.plots_dir}/")


if __name__ == "__main__":
    main()