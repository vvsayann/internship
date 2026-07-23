from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; just save PNG files
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


# ---------------------------------------------------------------------------
# 1. Line catalog: lines with occurrence >= 16/19 from the survey table
# ---------------------------------------------------------------------------

@dataclass
class SpectralLine:
    """Configuration for a single spectral line to search for and fit."""
    name: str
    rest_wavelength: float   # Angstrom
    window: float = 15.0     # Angstrom, half-width of the search window


# Only lines that appeared in >= 16 of the 19 reference spectra.
LINE_CATALOG: list[SpectralLine] = [
    SpectralLine("He I 3818",   3818.0, window=12.0),
    SpectralLine("H8 + He I 3890", 3890.0, window=14.0),
    SpectralLine("He I 4028",   4028.0, window=12.0),
    SpectralLine("He I 4473",   4473.0, window=12.0),
    SpectralLine("He I 4715",   4715.0, window=12.0),
    SpectralLine("He I 4925",   4925.0, window=12.0),
]


# ---------------------------------------------------------------------------
# 2. Result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Standardized output of a single line fit."""
    file_name: str
    line_name: str
    rest_wavelength: float
    fit_type: str = "none"        # "gaussian", "sigmoid", or "none"
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


# ---------------------------------------------------------------------------
# 3. SpectrumReader: FITS I/O only
# ---------------------------------------------------------------------------

class SpectrumReader:
    """Reads an SDSS-style spectrum FITS file into wavelength/flux arrays."""

    @staticmethod
    def load(filepath: str) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Returns (wavelength, flux, ivar).
        wavelength is in Angstrom (converted from loglam if needed).
        ivar (inverse variance) is returned if present, else None.
        """
        with fits.open(filepath) as hdul:
            data = None
            # SDSS spec files store the spectrum table in HDU 1 (COADD),
            # with columns typically named FLUX, LOGLAM, IVAR (or WAVELENGTH).
            for hdu in hdul[1:]:
                if hasattr(hdu, "columns") and hdu.data is not None:
                    colnames = [c.upper() for c in hdu.columns.names]
                    if "FLUX" in colnames and ("LOGLAM" in colnames or "WAVELENGTH" in colnames):
                        data = hdu.data
                        colmap = {c.upper(): c for c in hdu.columns.names}
                        break

            if data is None:
                raise ValueError(
                    f"Could not find a FLUX/LOGLAM (or WAVELENGTH) table in {filepath}"
                )

            flux = np.asarray(data[colmap["FLUX"]], dtype=float)

            if "LOGLAM" in colmap:
                wavelength = 10.0 ** np.asarray(data[colmap["LOGLAM"]], dtype=float)
            else:
                wavelength = np.asarray(data[colmap["WAVELENGTH"]], dtype=float)

            ivar = None
            if "IVAR" in colmap:
                ivar = np.asarray(data[colmap["IVAR"]], dtype=float)

        # Sort by wavelength just in case, and drop non-finite values.
        good = np.isfinite(wavelength) & np.isfinite(flux)
        wavelength, flux = wavelength[good], flux[good]
        if ivar is not None:
            ivar = ivar[good]
        order = np.argsort(wavelength)
        wavelength, flux = wavelength[order], flux[order]
        if ivar is not None:
            ivar = ivar[order]

        return wavelength, flux, ivar


# ---------------------------------------------------------------------------
# 4. ContinuumNormalizer: flatten the continuum to ~1.0
# ---------------------------------------------------------------------------

class ContinuumNormalizer:
    """Estimates and removes the stellar continuum so lines sit on a flat baseline."""

    @staticmethod
    def normalize(
        wavelength: np.ndarray,
        flux: np.ndarray,
        poly_degree: int = 5,
        medfilt_kernel: int = 51,
    ) -> np.ndarray:
        """
        Two-stage continuum removal:
          1. Median-filter the spectrum to smooth over narrow absorption dips.
          2. Fit a low-order polynomial to that smoothed curve as the continuum.
        Returns flux_norm = flux / continuum (line-free regions ~ 1.0).
        """
        kernel = medfilt_kernel if medfilt_kernel % 2 == 1 else medfilt_kernel + 1
        kernel = min(kernel, len(flux) - (1 - len(flux) % 2))
        kernel = max(kernel, 3)

        smoothed = medfilt(flux, kernel_size=kernel)

        coeffs = np.polyfit(wavelength, smoothed, poly_degree)
        continuum = np.polyval(coeffs, wavelength)
        continuum[continuum <= 0] = np.nanmedian(flux[flux > 0]) if np.any(flux > 0) else 1.0

        flux_norm = flux / continuum
        return flux_norm


# ---------------------------------------------------------------------------
# 5. LineFitter: Gaussian + sigmoid dip models
# ---------------------------------------------------------------------------

def gaussian_dip(x, amp, cen, sigma, offset):
    """Symmetric absorption dip."""
    return offset - amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))


def sigmoid_dip(x, amp, cen, width, offset):
    """Asymmetric step-like dip, useful for blended lines (e.g. H8 + He I)."""
    return offset - amp / (1 + np.exp((x - cen) / width))


class LineFitter:
    """Fits a Gaussian and a sigmoid model to a windowed region and keeps the better one."""

    def __init__(self, min_points: int = 8):
        self.min_points = min_points

    @staticmethod
    def _r_squared(y_obs: np.ndarray, y_fit: np.ndarray) -> float:
        ss_res = np.sum((y_obs - y_fit) ** 2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def fit_gaussian(self, x: np.ndarray, y: np.ndarray, guess_center: float):
        amp0 = max(np.max(y) - np.min(y), 0.01)
        sigma0 = max((x.max() - x.min()) / 6.0, 0.5)
        offset0 = float(np.max(y))
        p0 = [amp0, guess_center, sigma0, offset0]
        bounds = (
            [0, x.min(), 0.1, 0.0],
            [10 * amp0 + 1e-6, x.max(), (x.max() - x.min()), 10 * offset0 + 1e-6],
        )
        popt, _ = curve_fit(gaussian_dip, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_fit = gaussian_dip(x, *popt)
        return popt, self._r_squared(y, y_fit)

    def fit_sigmoid(self, x: np.ndarray, y: np.ndarray, guess_center: float):
        amp0 = max(np.max(y) - np.min(y), 0.01)
        width0 = max((x.max() - x.min()) / 8.0, 0.5)
        offset0 = float(np.max(y))
        p0 = [amp0, guess_center, width0, offset0]
        bounds = (
            [0, x.min(), 0.05, 0.0],
            [10 * amp0 + 1e-6, x.max(), (x.max() - x.min()), 10 * offset0 + 1e-6],
        )
        popt, _ = curve_fit(sigmoid_dip, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_fit = sigmoid_dip(x, *popt)
        return popt, self._r_squared(y, y_fit)

    def fit_line(
        self,
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        line: SpectralLine,
        file_name: str,
    ) -> FitResult:
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

        gauss_result, sigmoid_result = None, None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                popt_g, r2_g = self.fit_gaussian(x, y, line.rest_wavelength)
                gauss_result = ("gaussian", popt_g, r2_g)
            except Exception:
                pass

            try:
                popt_s, r2_s = self.fit_sigmoid(x, y, line.rest_wavelength)
                sigmoid_result = ("sigmoid", popt_s, r2_s)
            except Exception:
                pass

        candidates = [r for r in (gauss_result, sigmoid_result) if r is not None]
        if not candidates:
            return FitResult(
                file_name=file_name,
                line_name=line.name,
                rest_wavelength=line.rest_wavelength,
                note="both fits failed to converge",
            )

        # Pick whichever model achieved the higher R^2.
        fit_type, popt, r2 = max(candidates, key=lambda r: r[2])
        amp, cen, width_param, offset = popt

        return FitResult(
            file_name=file_name,
            line_name=line.name,
            rest_wavelength=line.rest_wavelength,
            fit_type=fit_type,
            center=float(cen),
            depth=float(amp),
            width=float(width_param),
            r_squared=float(r2),
            success=True,
            note="",
        )


# ---------------------------------------------------------------------------
# 6. SpectrumPlotter: saves a PNG showing the full spectrum + zoomed fit panels
# ---------------------------------------------------------------------------

class SpectrumPlotter:
    """Renders one PNG per spectrum: full normalized spectrum on top, and a
    zoomed-in panel per catalog line underneath showing the data + fitted curve."""

    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot(
        self,
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        line_catalog: list[SpectralLine],
        results: list[FitResult],
        file_name: str,
    ) -> str:
        n_lines = len(line_catalog)
        n_cols = 3
        n_rows = int(np.ceil(n_lines / n_cols))

        fig = plt.figure(figsize=(14, 4 + 3 * n_rows))
        gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[2] + [1] * n_rows)

        # --- top panel: full normalized spectrum ---
        ax_full = fig.add_subplot(gs[0, :])
        ax_full.plot(wavelength, flux_norm, color="black", lw=0.6)
        for line in line_catalog:
            ax_full.axvline(line.rest_wavelength, color="tab:red", ls="--", lw=0.7, alpha=0.6)
        ax_full.set_title(f"{file_name} \u2014 full normalized spectrum")
        ax_full.set_xlabel("Wavelength (\u00c5)")
        ax_full.set_ylabel("Normalized flux")

        # --- bottom panels: zoomed fit per line ---
        for i, (line, result) in enumerate(zip(line_catalog, results)):
            row, col = divmod(i, n_cols)
            ax = fig.add_subplot(gs[row + 1, col])

            lo, hi = line.rest_wavelength - line.window, line.rest_wavelength + line.window
            mask = (wavelength >= lo) & (wavelength <= hi)
            x, y = wavelength[mask], flux_norm[mask]
            ax.plot(x, y, "o", ms=3, color="black", label="data")

            if result.success:
                x_fine = np.linspace(x.min(), x.max(), 200)
                if result.fit_type == "gaussian":
                    y_fine = gaussian_dip(x_fine, result.depth, result.center,
                                          result.width, np.max(y))
                else:
                    y_fine = sigmoid_dip(x_fine, result.depth, result.center,
                                         result.width, np.max(y))
                ax.plot(x_fine, y_fine, "-", color="tab:red", lw=1.5,
                        label=f"{result.fit_type} fit")
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


# ---------------------------------------------------------------------------
# 7. SpectrumProcessor: glue for a single file
# ---------------------------------------------------------------------------

class SpectrumProcessor:
    def __init__(
        self,
        line_catalog: list[SpectralLine],
        fitter: Optional[LineFitter] = None,
        plotter: Optional[SpectrumPlotter] = None,
    ):
        self.line_catalog = line_catalog
        self.fitter = fitter or LineFitter()
        self.plotter = plotter  # None => no plots generated

    def process(self, filepath: str) -> list[FitResult]:
        file_name = os.path.basename(filepath)
        wavelength, flux, _ivar = SpectrumReader.load(filepath)
        flux_norm = ContinuumNormalizer.normalize(wavelength, flux)

        results = []
        for line in self.line_catalog:
            result = self.fitter.fit_line(wavelength, flux_norm, line, file_name)
            results.append(result)

        if self.plotter is not None:
            png_path = self.plotter.plot(wavelength, flux_norm, self.line_catalog, results, file_name)
            print(f"    Saved plot: {png_path}")

        return results


# ---------------------------------------------------------------------------
# 8. BatchRunner: processes files one by one, saves running results table
# ---------------------------------------------------------------------------

class BatchRunner:
    def __init__(self, processor: Optional[SpectrumProcessor] = None):
        self.processor = processor or SpectrumProcessor(LINE_CATALOG)
        self.all_results: list[FitResult] = []

    def run(self, filepaths: list[str], out_csv: str = "line_fit_results.csv") -> pd.DataFrame:
        for i, filepath in enumerate(filepaths, start=1):
            print(f"[{i}/{len(filepaths)}] Processing {os.path.basename(filepath)} ...")
            try:
                results = self.processor.process(filepath)
                self.all_results.extend(results)
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

            # Save incrementally so a crash mid-batch doesn't lose earlier work.
            self._save(out_csv)

        return self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.all_results])

    def _save(self, out_csv: str) -> None:
        df = self._to_dataframe()
        df.to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
# 9. CLI entry point
# ---------------------------------------------------------------------------

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