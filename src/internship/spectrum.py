from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import medfilt
from .spectral_line import SpectralLine
from .fit_result import FitResult
from .fitters import LineFitter
from .utilities import sigmoid_dip , gaussian_dip, voigt_dip


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


class SpectrumPlotter:
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    @staticmethod
    def _build_total_fit(wavelength, continuum, line_catalog, results):
        """Continuum everywhere, but multiplied by the fitted line profile
        inside each successfully-fit line's window, so the black curve dips
        along with the real absorption lines instead of staying flat there."""
        total_fit = continuum.copy()
        for line, result in zip(line_catalog, results):
            if not result.success:
                continue

            lo, hi = line.rest_wavelength - line.window, line.rest_wavelength + line.window
            mask = (wavelength >= lo) & (wavelength <= hi)
            x = wavelength[mask]
            if x.size == 0:
                continue

            offset = result.continuum_offset if np.isfinite(result.continuum_offset) else 1.0
            if result.fit_type == "voigt":
                y_dip = voigt_dip(x, result.depth, result.center, result.width, result.gamma, offset)
            elif result.fit_type == "gaussian":
                y_dip = gaussian_dip(x, result.depth, result.center, result.width, offset)
            elif result.fit_type == "sigmoid":
                y_dip = sigmoid_dip(x, result.depth, result.center, result.width, offset)
            else:
                continue

            total_fit[mask] = continuum[mask] * y_dip
        return total_fit

    def plot(self, wavelength, flux, continuum, flux_norm, line_catalog, results, file_name) -> str:
        n_lines = len(line_catalog)
        n_cols = 3
        n_rows = int(np.ceil(n_lines / n_cols))

        fig = plt.figure(figsize=(14, 4 + 3 * n_rows))
        gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[2] + [1] * n_rows)

        total_fit = self._build_total_fit(wavelength, continuum, line_catalog, results)

        # --- Top panel: raw spectrum + total fit (continuum + line dips) ---
        ax_full = fig.add_subplot(gs[0, :])
        ax_full.plot(wavelength, flux, color="tab:blue", lw=0.6, label="Extinction spectrum")
        ax_full.plot(wavelength, total_fit, color="black", lw=1.2, label="Total fit")
        for line in line_catalog:
            ax_full.axvline(line.rest_wavelength, color="tab:red", ls="--", lw=0.7, alpha=0.6)
        ax_full.set_title(f"{file_name} \u2014 full spectrum with continuum fit")
        ax_full.set_xlabel("Wavelength (\u00c5)")
        ax_full.set_ylabel("Flux")
        ax_full.legend(fontsize=8)

        # --- Sub-panels: per-line fits, still on normalized flux ---
        for i, (line, result) in enumerate(zip(line_catalog, results)):
            row, col = divmod(i, n_cols)
            ax = fig.add_subplot(gs[row + 1, col])

            lo, hi = line.rest_wavelength - line.window, line.rest_wavelength + line.window
            mask = (wavelength >= lo) & (wavelength <= hi)
            x, y = wavelength[mask], flux_norm[mask]
            ax.plot(x, y, "o", ms=3, color="black", label="data")

            if result.success:
                x_fine = np.linspace(x.min(), x.max(), 200)
                offset = result.continuum_offset if np.isfinite(result.continuum_offset) else np.max(y)
                if result.fit_type == "voigt":
                    y_fine = voigt_dip(x_fine, result.depth, result.center, result.width, result.gamma, offset)
                elif result.fit_type == "gaussian":
                    y_fine = gaussian_dip(x_fine, result.depth, result.center, result.width, offset)
                elif result.fit_type == "sigmoid":
                    y_fine = sigmoid_dip(x_fine, result.depth, result.center, result.width, offset)
                else:
                    y_fine = None

                if y_fine is not None:
                    ax.plot(x_fine, y_fine, "-", color="tab:red", lw=1.5, label=f"{result.fit_type} fit")
                    ax.axvline(result.center, color="tab:blue", ls=":", lw=1)
                title = (
                    f"{line.name}\ncenter={result.center:.1f}\u00c5  "
                    f"R\u00b2={result.r_squared:.3f}  fRMS={result.fractional_rms:.3f}"
                )
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


LINE_CATALOG: list[SpectralLine] = [
    SpectralLine("He I 3818", 3818.0, window=12.0),
    SpectralLine("H8 + He I 3890", 3890.0, window=14.0),
    SpectralLine("He I 4028", 4028.0, window=12.0),
    SpectralLine("He I 4473", 4473.0, window=12.0),
    SpectralLine("He I 4715", 4715.0, window=12.0),
    SpectralLine("He I 4925", 4925.0, window=12.0),
    SpectralLine("He I 5015", 5015.7, window=12.0),
    SpectralLine("He I 5876", 5875.6, window=14.0),
    SpectralLine("He I 6678", 6678.15, window=12.0),
]


class SpectrumProcessor:
    def __init__(
        self,
        line_catalog: list[SpectralLine],
        fitter: Optional[LineFitter] = None,
        plotter: Optional[SpectrumPlotter] = None,
        min_r2: float = 0.0,
        allowed_fit_types: Optional[set[str]] = None,
    ):
        self.line_catalog = line_catalog
        self.fitter = fitter or LineFitter()
        self.plotter = plotter
        self.min_r2 = min_r2
        self.allowed_fit_types = allowed_fit_types

    def process(self, filepath: str) -> list[FitResult]:
        file_name = os.path.basename(filepath)
        wavelength, flux, _ivar = SpectrumReader.load(filepath)
        flux_norm, continuum = ContinuumNormalizer.normalize(wavelength, flux)

        all_results = [self.fitter.fit_line(wavelength, flux_norm, line, file_name) for line in self.line_catalog]

        kept_lines, kept_results = [], []
        for line, result in zip(self.line_catalog, all_results):
            if not result.success or result.r_squared is None or result.r_squared < self.min_r2:
                continue
            if self.allowed_fit_types is not None and result.fit_type not in self.allowed_fit_types:
                continue
            kept_lines.append(line)
            kept_results.append(result)

        if self.plotter is not None:
            if kept_results:
                png_path = self.plotter.plot(
                    wavelength, flux, continuum, flux_norm, kept_lines, kept_results, file_name
                )
                print(f"    Saved plot: {png_path}")
            else:
                print(f"    No lines matched filters for {file_name} \u2014 skipping plot")

        return all_results


class ContinuumNormalizer:
    @staticmethod
    def normalize(
        wavelength: np.ndarray,
        flux: np.ndarray,
        poly_degree: int = 5,
        medfilt_kernel: int = 51,
    ) -> tuple[np.ndarray, np.ndarray]:
        kernel = medfilt_kernel if medfilt_kernel % 2 == 1 else medfilt_kernel + 1
        kernel = min(kernel, len(flux) - (1 - len(flux) % 2))
        kernel = max(kernel, 3)

        smoothed = medfilt(flux, kernel_size=kernel)

        coeffs = np.polyfit(wavelength, smoothed, poly_degree)
        continuum = np.polyval(coeffs, wavelength)
        continuum[continuum <= 0] = np.nanmedian(flux[flux > 0]) if np.any(flux > 0) else 1.0

        return flux / continuum, continuum


class BatchRunner:

    def __init__(self, processor: Optional[SpectrumProcessor] = None):
        self.processor = processor or SpectrumProcessor(LINE_CATALOG)
        self.all_results: list[FitResult] = []

    def run(self, filepaths: list[str], out_dir: str = ".") -> pd.DataFrame:
        os.makedirs(out_dir, exist_ok=True)

        for i, filepath in enumerate(filepaths, start=1):
            file_name = os.path.basename(filepath)
            print(f"[{i}/{len(filepaths)}] Processing {file_name} ...")
            try:
                file_results = self.processor.process(filepath)
            except Exception as e:
                print(f"    !! Failed to process {filepath}: {e}")
                file_results = [
                    FitResult(
                        file_name=file_name,
                        line_name="ALL",
                        rest_wavelength=np.nan,
                        note=f"file-level error: {e}",
                    )
                ]

            self.all_results.extend(file_results)
            self._save_file_results(file_results, file_name, out_dir)

        return self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.all_results])

    def _save_file_results(self, results: list[FitResult], file_name: str, out_dir: str) -> str:
        base = os.path.splitext(file_name)[0]

        fit_types = [r.fit_type for r in results if r.success]
        if fit_types:
            dominant = max(set(fit_types), key=fit_types.count)
        else:
            dominant = "nofit"

        out_path = os.path.join(out_dir, f"{dominant}_{base}.csv")
        pd.DataFrame([r.to_dict() for r in results]).to_csv(out_path, index=False)
        print(f"    Saved results: {out_path}")
        return out_path