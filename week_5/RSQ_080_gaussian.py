from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")

from src.internship.fitters import LineFitter
from src.internship.spectrum import BatchRunner, SpectrumPlotter, LINE_CATALOG, SpectrumProcessor

SPECTRA_FOLDER = r"C:\Users\Ayank\OneDrive\Desktop\internship\spectra_files"
INPUT_FOLDER = os.path.join(SPECTRA_FOLDER, "processed_spectrum")
OUT_DIR = os.path.join(SPECTRA_FOLDER, "gaussian_fit_results_r080")
MIN_R2 = 0.80


def main():
    filepaths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.fits")))
    if not filepaths:
        print(f"No FITS files found in {INPUT_FOLDER}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    plotter = SpectrumPlotter(OUT_DIR)
    fitter = LineFitter(verbose=False, allowed_fit_types={"gaussian"})
    processor = SpectrumProcessor(
        LINE_CATALOG,
        fitter=fitter,
        plotter=plotter,
        min_r2=MIN_R2,
        allowed_fit_types={"gaussian"},
    )
    runner = BatchRunner(processor=processor)
    df = runner.run(filepaths, out_dir=OUT_DIR)

    print("\n=== Summary (Gaussian, R^2 >= 0.80) ===")
    print(f"Files processed : {df['file_name'].nunique()}")
    print(f"Line fits kept  : {len(df)}")
    print(f"Results saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()