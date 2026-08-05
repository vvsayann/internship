from __future__ import annotations

import argparse
import glob
import os
import sys

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from scipy.special import wofz

from src.internship.fit_result import FitResult
from src.internship.fitters import LineFitter

from src.internship.spectrum import SpectrumReader, SpectrumPlotter, SpectralLine, LINE_CATALOG, SpectrumProcessor, \
    BatchRunner, ContinuumNormalizer

from src.internship.utilities import voigt_dip

try:
    from astropy.io import fits
except ImportError as e:
    raise ImportError(
        "astropy is required to read FITS files. Install with: "
        "pip install astropy --break-system-packages"
    ) from e


def _voigt_peak_normalized(x, cen, sigma, gamma):
    sigma = max(sigma, 1e-6)
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z))
    peak = np.real(wofz(1j * gamma / (sigma * np.sqrt(2))))
    peak = peak if peak > 1e-12 else 1e-12
    return profile / peak


def voigt_double_dip(x, amp1, cen1, sigma1, gamma1, amp2, cen2, sigma2, gamma2, offset):
    return (
            offset
            - amp1 * _voigt_peak_normalized(x, cen1, sigma1, gamma1)
            - amp2 * _voigt_peak_normalized(x, cen2, sigma2, gamma2)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically fit recurring spectral lines (occurrence >= 16/19) "
                    "in one or more SDSS FITS spectra using a Voigt profile."
    )
    parser.add_argument("files", nargs="*", help="Path(s) to FITS file(s)")
    parser.add_argument("--folder", type=str, default=None,
                        help="Folder containing .fits files to process (all files in it)")
    parser.add_argument("--out", type=str, default="line_fit_results.csv",
                        help="Output CSV path (default: line_fit_results.csv)")
    parser.add_argument("--plots-dir", type=str, default=None,
                        help="If given, save a PNG per file (spectrum + fitted line panels) "
                             "into this folder. Omit this flag to skip plotting.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the R^2 (and sigma/gamma) of the Voigt fit for every line.")
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
    fitter = LineFitter(verbose=args.verbose)
    processor = SpectrumProcessor(LINE_CATALOG, fitter=fitter, plotter=plotter)
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
