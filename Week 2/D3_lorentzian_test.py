from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import medfilt

try:
    from astropy.io import fits
except ImportError as e:
    raise ImportError(
        "astropy is required to read FITS files. Install with: "
        "pip install astropy --break-system-packages"
    ) from e


@dataclass
class SpectralLine:
    name: str
    rest_wavelength: float
    window: float = 15.0


LINE_CATALOG: list[SpectralLine] = [
    SpectralLine("He I 3818", 3818.0, window=12.0),
    SpectralLine("H8 + He I 3890", 3890.0, window=14.0),
    SpectralLine("He I 4028", 4028.0, window=12.0),
    SpectralLine("He I 4473", 4473.0, window=12.0),
    SpectralLine("He I 4715", 4715.0, window=12.0),
    SpectralLine("He I 4925", 4925.0, window=12.0),
]


@dataclass
class FitResult:
    file_name: str
    line_name: str
    rest_wavelength: float
    fit_type: str = "none"
    center: float = np.nan
    depth: float = np.nan
    width: float = np.nan
    r_squared: float = np.nan
    success: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "line_name": self.line_name,
            "rest_wavelength": self.rest_wavelength,
            "fit_type": self.fit_type,
            "fitted_center": self.center,
            "depth": self.depth,
            "width": self.width,
            "r_squared": self.r_squared,
            "success": self.success,
            "note": self.note,
        }


class SpectrumReader:
    @staticmethod
    def load(filepath: str) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        with fits.open(filepath) as hdul:
            data = None
            colmap = None
            for hdu in hdul[1:]:
                if hasattr(hdu, "columns") and hdu.data is not None:
                    colnames = [c.upper() for c in hdu.columns.names]
                    if "FLUX" in colnames and ("LOGLAM" in colnames or "WAVELENGTH" in colnames):
                        data = hdu.data
                        colmap = {c.upper(): c for c in hdu.columns.names}
                        break

            if data is None:
                raise ValueError(f"Could not find a FLUX/LOGLAM (or WAVELENGTH) table in {filepath}")

            flux = np.asarray(data[colmap["FLUX"]], dtype=float)

            if "LOGLAM" in colmap:
                wavelength = 10.0 ** np.asarray(data[colmap["LOGLAM"]], dtype=float)
            else:
                wavelength = np.asarray(data[colmap["WAVELENGTH"]], dtype=float)

            ivar = None
            if "IVAR" in colmap:
                ivar = np.asarray(data[colmap["IVAR"]], dtype=float)

        good = np.isfinite(wavelength) & np.isfinite(flux)
        wavelength, flux = wavelength[good], flux[good]
        if ivar is not None:
            ivar = ivar[good]

        order = np.argsort(wavelength)
        wavelength, flux = wavelength[order], flux[order]
        if ivar is not None:
            ivar = ivar[order]

        return wavelength, flux, ivar


class ContinuumNormalizer:
    @staticmethod
    def normalize(
        wavelength: np.ndarray,
        flux: np.ndarray,
        poly_degree: int = 5,
        medfilt_kernel: int = 51,
    ) -> np.ndarray:
        kernel = medfilt_kernel if medfilt_kernel % 2 == 1 else medfilt_kernel + 1
        kernel = min(kernel, len(flux) - (1 - len(flux) % 2))
        kernel = max(kernel, 3)

        smoothed = medfilt(flux, kernel_size=kernel)

        coeffs = np.polyfit(wavelength, smoothed, poly_degree)
        continuum = np.polyval(coeffs, wavelength)
        continuum[continuum <= 0] = np.nanmedian(flux[flux > 0]) if np.any(flux > 0) else 1.0

        return flux / continuum


def lorentzian_dip(x, amp, cen, gamma, offset):
    return offset - amp * (gamma ** 2 / ((x - cen) ** 2 + gamma ** 2))


class LineFitter:
    def __init__(self, min_points: int = 8):
        self.min_points = min_points

    @staticmethod
    def _r_squared(y_obs: np.ndarray, y_fit: np.ndarray) -> float:
        ss_res = np.sum((y_obs - y_fit) ** 2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def fit_lorentzian(self, x: np.ndarray, y: np.ndarray, guess_center: float):
        amp0 = max(np.max(y) - np.min(y), 0.01)
        gamma0 = max((x.max() - x.min()) / 6.0, 0.5)
        offset0 = float(np.max(y))
        p0 = [amp0, guess_center, gamma0, offset0]
        bounds = (
            [0, x.min(), 0.1, 0.0],
            [10 * amp0 + 1e-6, x.max(), (x.max() - x.min()), 10 * offset0 + 1e-6],
        )
        popt, _ = curve_fit(lorentzian_dip, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return popt, self._r_squared(y, lorentzian_dip(x, *popt))

    def fit_line(self, wavelength: np.ndarray, flux_norm: np.ndarray, line: SpectralLine, file_name: str) -> FitResult:
        lo, hi = line.rest_wavelength - line.window, line.rest_wavelength + line.window
        mask = (wavelength >= lo) & (wavelength <= hi)
        x, y = wavelength[mask], flux_norm[mask]

        if len(x) < self.min_points:
            return FitResult(
                file_name=file_name,
                line_name=line.name,
                rest_wavelength=line.rest_wavelength,
                note="insufficient data points in window",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt, r2 = self.fit_lorentzian(x, y, line.rest_wavelength)
            except Exception as e:
                return FitResult(
                    file_name=file_name,
                    line_name=line.name,
                    rest_wavelength=line.rest_wavelength,
                    note=f"lorentzian fit failed to converge: {e}",
                )

        amp, cen, gamma, offset = popt

        return FitResult(
            file_name=file_name,
            line_name=line.name,
            rest_wavelength=line.rest_wavelength,
            fit_type="lorentzian",
            center=float(cen),
            depth=float(amp),
            width=float(gamma),
            r_squared=float(r2),
            success=True,
        )


class SpectrumPlotter:
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot(self, wavelength, flux_norm, line_catalog, results, file_name) -> str:
        n_lines = len(line_catalog)
        n_cols = 3
        n_rows = int(np.ceil(n_lines / n_cols))

        fig = plt.figure(figsize=(14, 4 + 3 * n_rows))
        gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[2] + [1] * n_rows)

        ax_full = fig.add_subplot(gs[0, :])
        ax_full.plot(wavelength, flux_norm, color="black", lw=0.6)
        for line in line_catalog:
            ax_full.axvline(line.rest_wavelength, color="tab:red", ls="--", lw=0.7, alpha=0.6)
        ax_full.set_title(f"{file_name} \u2014 full normalized spectrum")
        ax_full.set_xlabel("Wavelength (\u00c5)")
        ax_full.set_ylabel("Normalized flux")

        for i, (line, result) in enumerate(zip(line_catalog, results)):
            row, col = divmod(i, n_cols)
            ax = fig.add_subplot(gs[row + 1, col])

            lo, hi = line.rest_wavelength - line.window, line.rest_wavelength + line.window
            mask = (wavelength >= lo) & (wavelength <= hi)
            x, y = wavelength[mask], flux_norm[mask]
            ax.plot(x, y, "o", ms=3, color="black", label="data")

            if result.success:
                x_fine = np.linspace(x.min(), x.max(), 200)
                y_fine = lorentzian_dip(x_fine, result.depth, result.center, result.width, np.max(y))
                ax.plot(x_fine, y_fine, "-", color="tab:red", lw=1.5, label=f"{result.fit_type} fit")
                ax.axvline(result.center, color="tab:blue", ls=":", lw=1)
                title = f"{line.name}\ncenter={result.center:.1f}\u00c5  R\u00b2={result.r_squared:.3f}"
            else:
                title = f"{line.name}\nfit failed: {result.note}"

            ax.axvline(line.rest_wavelength, color="gray", ls="--", lw=0.7, alpha=0.6)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("\u00c5", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="lower right")

        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, os.path.splitext(file_name)[0] + "_fits.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return out_path


class SpectrumProcessor:
    def __init__(
        self,
        line_catalog: list[SpectralLine],
        fitter: Optional[LineFitter] = None,
        plotter: Optional[SpectrumPlotter] = None,
    ):
        self.line_catalog = line_catalog
        self.fitter = fitter or LineFitter()
        self.plotter = plotter

    def process(self, filepath: str) -> list[FitResult]:
        file_name = os.path.basename(filepath)
        wavelength, flux, _ivar = SpectrumReader.load(filepath)
        flux_norm = ContinuumNormalizer.normalize(wavelength, flux)

        results = [self.fitter.fit_line(wavelength, flux_norm, line, file_name) for line in self.line_catalog]

        if self.plotter is not None:
            png_path = self.plotter.plot(wavelength, flux_norm, self.line_catalog, results, file_name)
            print(f"    Saved plot: {png_path}")

        return results


class BatchRunner:
    def __init__(self, processor: Optional[SpectrumProcessor] = None):
        self.processor = processor or SpectrumProcessor(LINE_CATALOG)
        self.all_results: list[FitResult] = []

    def run(self, filepaths: list[str], out_csv: str = "line_fit_results.csv") -> pd.DataFrame:
        for i, filepath in enumerate(filepaths, start=1):
            print(f"[{i}/{len(filepaths)}] Processing {os.path.basename(filepath)} ...")
            try:
                self.all_results.extend(self.processor.process(filepath))
            except Exception as e:
                print(f"    !! Failed to process {filepath}: {e}")
                self.all_results.append(
                    FitResult(
                        file_name=os.path.basename(filepath),
                        line_name="ALL",
                        rest_wavelength=np.nan,
                        note=f"file-level error: {e}",
                    )
                )
            self._save(out_csv)

        return self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.all_results])

    def _save(self, out_csv: str) -> None:
        self._to_dataframe().to_csv(out_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically fit recurring spectral lines (occurrence >= 16/19) "
                    "in one or more SDSS FITS spectra using a Lorentzian profile."
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