from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
from src.internship.fitters import LineFitter
from src.internship.spectrum import BatchRunner
from src.internship.spectrum import SpectrumPlotter, LINE_CATALOG, SpectrumProcessor
from src.internship.utilities import voigt_dip, _voigt_peak_normalized

try:
    from astropy.io import fits
except ImportError as e:
    raise ImportError(
        "astropy is required to read FITS files. Install with: "
        "pip install astropy --break-system-packages"
    ) from e


SPECTRA_FOLDER = r"C:\Users\Ayank\OneDrive\Desktop\internship\spectra_files"
DEFAULT_INPUT_FOLDER = os.path.join(SPECTRA_FOLDER, "processed_spectrum")
DEFAULT_OUT_DIR = os.path.join(SPECTRA_FOLDER, "voigt_fit_results")
DEFAULT_PLOTS_DIR = DEFAULT_OUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically fit recurring spectral lines (occurrence >= 16/19) "
                    "in one or more SDSS FITS spectra using a Voigt profile."
    )
    parser.add_argument("files", nargs="*", help="Path(s) to FITS file(s)")
    parser.add_argument("--folder", type=str, default=DEFAULT_INPUT_FOLDER,
                        help="Folder containing .fits files to process (all files in it)")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Directory to write per-file 'voigt_<filename>.csv' results into "
                             f"(default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--plots-dir", type=str, default=DEFAULT_PLOTS_DIR,
                        help="Folder to save a PNG per file (spectrum + fitted line panels) into. "
                             f"(default: {DEFAULT_PLOTS_DIR})")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plotting entirely.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the R^2 (and sigma/gamma) of the Voigt fit for every line.")
    return parser.parse_args()


def main():
    args = parse_args()

    filepaths = list(args.files)
    if args.folder:
        filepaths.extend(sorted(glob.glob(os.path.join(args.folder, "*.fits"))))

    if not filepaths:
        print(f"No FITS files given and none found in {args.folder}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    plots_dir = None if args.no_plots else args.plots_dir
    plotter = SpectrumPlotter(plots_dir) if plots_dir else None
    fitter = LineFitter(verbose=args.verbose)
    processor = SpectrumProcessor(LINE_CATALOG, fitter=fitter, plotter=plotter)
    runner = BatchRunner(processor=processor)
    df = runner.run(filepaths, out_dir=args.out_dir)

    print("\n=== Summary ===")
    print(f"Files processed : {df['file_name'].nunique()}")
    print(f"Total line fits : {len(df)}")
    print(f"Successful fits : {int(df['success'].sum())}")
    print(f"Results saved to: {args.out_dir}/voigt_<filename>.csv (one CSV per input file)")
    if plots_dir:
        print(f"Plots saved to  : {plots_dir}/")


if __name__ == "__main__":
    main()