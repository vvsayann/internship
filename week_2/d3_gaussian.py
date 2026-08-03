from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

from src.internship.spectrum import SpectrumPlotter, SpectrumProcessor, LINE_CATALOG, BatchRunner
from src.internship.fitters import LineFitter

matplotlib.use("Agg")

try:
    from astropy.io import fits
except ImportError as e:
    raise ImportError(
        "astropy is required to read FITS files. Install with: "
        "pip install astropy --break-system-packages"
    ) from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically fit recurring spectral lines in one or more SDSS FITS "
                    "spectra using a pure Gaussian profile."
    )
    parser.add_argument("files", nargs="*", help="Path(s) to FITS file(s)")
    parser.add_argument("--folder", type=str, default=None,
                        help="Folder containing .fits files to process (all files in it)")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory to write per-file 'gaussian_<filename>.csv' results into "
                             "(default: current directory)")
    parser.add_argument("--plots-dir", type=str, default=None,
                        help="If given, save a PNG per file (spectrum + fitted line panels) "
                             "into this folder. Omit this flag to skip plotting.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the R^2 (and sigma) of the Gaussian fit for every line.")
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
    df = runner.run(filepaths, out_dir=args.out_dir)

    print("\n=== Summary ===")
    print(f"Files processed : {df['file_name'].nunique()}")
    print(f"Total line fits : {len(df)}")
    print(f"Successful fits : {int(df['success'].sum())}")
    print(f"Results saved to: {args.out_dir}/gaussian_<filename>.csv (one CSV per input file)")
    if args.plots_dir:
        print(f"Plots saved to  : {args.plots_dir}/")


if __name__ == "__main__":
    main()