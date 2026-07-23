"""
spectral_line_fitter_voigt.py

Automatic spectral line fitting pipeline for SDSS OB-star FITS spectra.
This version fits a Voigt profile only (Gaussian core + Lorentzian wings --
the physically realistic profile for most stellar absorption lines).

Most lines are fit with a single Voigt dip. The He I 4028 line is fit with
TWO overlapping Voigt dips (a "double-dip" fit), since visual inspection
showed two blended features there that a single symmetric profile could not
represent well. To mark any other line as blended, set n_components=2 on its
SpectralLine entry in LINE_CATALOG below.

Pipeline:
    BatchRunner
      -> for each fits_file:
           SpectrumProcessor.process()
             -> SpectrumReader.load()
             -> ContinuumNormalizer.normalize()
             -> LineFitter.fit_line()  for every line in LINE_CATALOG
                  fits 1 Voigt dip (or 2, for blended lines)
           -> results appended to a running table
      -> results saved to CSV

Usage:
    python spectral_line_fitter_voigt.py file1.fits file2.fits ...
    python spectral_line_fitter_voigt.py --folder /path/to/fits_dir
    python spectral_line_fitter_voigt.py --folder /path/to/fits_dir --out results.csv
    python spectral_line_fitter_voigt.py --folder /path/to/fits_dir --plots-dir plots/

--plots-dir is optional: if given, a PNG is saved per file showing the full
normalized spectrum plus a zoomed-in panel per catalog line with the fitted
curve overlaid, so you can visually check each fit. Without it, only the
CSV table of numeric results is produced.

Dependencies: numpy, scipy, pandas, astropy, matplotlib
"""

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
matplotlib.use("Agg")  # no display needed; just save PNG files
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy.special import wofz

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
    n_components: int = 1    # set to 2 for a blended/double-dip feature


# Only lines that appeared in >= 16 of the 19 reference spectra.
# He I 4028 is marked as a 2-component (blended) fit based on visual inspection
# showing two overlapping dips rather than one clean symmetric feature.
LINE_CATALOG: list[SpectralLine] = [
    SpectralLine("He I 3818",   3818.0, window=12.0),
    SpectralLine("H8 + He I 3890", 3890.0, window=14.0),
    SpectralLine("He I 4028",   4028.0, window=16.0, n_components=2),
    SpectralLine("He I 4473",   4473.0, window=12.0),
    SpectralLine("He I 4715",   4715.0, window=12.0),
    SpectralLine("He I 4925",   4925.0, window=12.0),
]


# ---------------------------------------------------------------------------
# 2. Result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Standardized output of a single line fit (or one component of a
    2-component blended fit)."""
    file_name: str
    line_name: str
    rest_wavelength: float
    fit_type: str = "none"        # "voigt" or "voigt_double"
    component: str = ""           # "" for single fits, "1" or "2" for double-dip components
    center: float = np.nan
    depth: float = np.nan
    width: float = np.nan         # sigma, the Gaussian component of the Voigt profile
    extra_width: float = np.nan   # gamma, the Lorentzian component of the Voigt profile
    r_squared: float = np.nan
    success: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "line_name": self.line_name,
            "rest_wavelength": self.rest_wavelength,
            "fit_type": self.fit_type,
            "component": self.component,
            "fitted_center": self.center,
            "depth": self.depth,
            "sigma_gaussian_component": self.width,
            "gamma_lorentzian_component": self.extra_width,
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
    """Estimates and removes the stellar continuum so lines sit on a flat baseline.

    Deep or broad absorption lines can drag a naive polynomial-through-the-
    median-filter continuum downward, which under-corrects the line depth
    and can bias the normalized spectrum near that line. To guard against
    this, the continuum polynomial is fit iteratively: after each fit we
    flag points that sit far *below* the current continuum estimate (i.e.
    line cores) and exclude them from the next fit. Only low-side outliers
    are clipped (absorption dips) -- not high-side ones -- since symmetric
    clipping would also start discarding legitimate continuum points.
    """

    @staticmethod
    def normalize(
        wavelength: np.ndarray,
        flux: np.ndarray,
        poly_degree: int = 5,
        medfilt_kernel: int = 51,
        sigma_clip: float = 2.5,
        max_iter: int = 5,
    ) -> np.ndarray:
        """
        Continuum removal:
          1. Median-filter the spectrum to smooth over narrow absorption dips.
          2. Iteratively fit a low-order polynomial to that smoothed curve,
             each round masking out points that fall > sigma_clip standard
             deviations *below* the current fit (deep/broad line cores),
             and refitting on the surviving points only.
          3. Evaluate the final polynomial at every wavelength as the
             continuum, so flux_norm has full wavelength coverage even
             though the fit itself ignored line cores.
        Returns flux_norm = flux / continuum (line-free regions ~ 1.0).
        """
        kernel = medfilt_kernel if medfilt_kernel % 2 == 1 else medfilt_kernel + 1
        kernel = min(kernel, len(flux) - (1 - len(flux) % 2))
        kernel = max(kernel, 3)

        smoothed = medfilt(flux, kernel_size=kernel)

        min_pts = poly_degree + 2
        mask = np.ones(len(smoothed), dtype=bool)
        coeffs = np.polyfit(wavelength, smoothed, poly_degree)

        for _ in range(max_iter):
            continuum_est = np.polyval(coeffs, wavelength)
            resid = smoothed - continuum_est
            # Robust spread estimate on the currently-kept points.
            std = np.std(resid[mask])
            if std == 0 or not np.isfinite(std):
                break

            new_mask = resid > -sigma_clip * std  # keep everything except deep dips
            if new_mask.sum() < min_pts:
                # Clipping got too aggressive (e.g. a very broad/deep line
                # dominating the window) -- stop before we run out of points.
                break
            if np.array_equal(new_mask, mask):
                break  # converged

            mask = new_mask
            coeffs = np.polyfit(wavelength[mask], smoothed[mask], poly_degree)

        continuum = np.polyval(coeffs, wavelength)
        continuum[continuum <= 0] = np.nanmedian(flux[flux > 0]) if np.any(flux > 0) else 1.0

        flux_norm = flux / continuum
        return flux_norm


# ---------------------------------------------------------------------------
# 5. LineFitter: Voigt dip model
# ---------------------------------------------------------------------------

def _voigt_peak_normalized(x, cen, sigma, gamma):
    """Voigt profile (Gaussian convolved with Lorentzian), normalized so the
    peak value at x = cen is exactly 1.0 (so 'amp' below means true dip depth)."""
    sigma = max(sigma, 1e-6)
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z))
    peak = np.real(wofz(1j * gamma / (sigma * np.sqrt(2))))
    peak = peak if peak > 1e-12 else 1e-12
    return profile / peak


def voigt_dip(x, amp, cen, sigma, gamma, offset):
    """Voigt absorption dip: combines a Gaussian core (sigma) with Lorentzian
    wings (gamma) -- the physically realistic profile for most stellar lines."""
    return offset - amp * _voigt_peak_normalized(x, cen, sigma, gamma)


def voigt_double_dip(x, amp1, cen1, sigma1, gamma1, amp2, cen2, sigma2, gamma2, offset):
    """Two independent Voigt dips sharing one continuum offset -- for a
    blended/overlapping pair of lines that a single profile can't fit well."""
    return (
        offset
        - amp1 * _voigt_peak_normalized(x, cen1, sigma1, gamma1)
        - amp2 * _voigt_peak_normalized(x, cen2, sigma2, gamma2)
    )


class LineFitter:
    """Fits a Voigt model (single or 2-component blended) to a windowed
    region around each line."""

    def __init__(self, min_points: int = 8, verbose: bool = False):
        self.min_points = min_points
        self.verbose = verbose

    @staticmethod
    def _r_squared(y_obs: np.ndarray, y_fit: np.ndarray) -> float:
        ss_res = np.sum((y_obs - y_fit) ** 2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def fit_voigt(self, x: np.ndarray, y: np.ndarray, guess_center: float):
        amp0 = max(np.max(y) - np.min(y), 0.01)
        sigma0 = max((x.max() - x.min()) / 8.0, 0.5)
        gamma0 = max((x.max() - x.min()) / 8.0, 0.5)
        offset0 = float(np.max(y))
        p0 = [amp0, guess_center, sigma0, gamma0, offset0]
        bounds = (
            [0, x.min(), 0.05, 0.0, 0.0],
            [10 * amp0 + 1e-6, x.max(), (x.max() - x.min()), (x.max() - x.min()), 10 * offset0 + 1e-6],
        )
        popt, _ = curve_fit(voigt_dip, x, y, p0=p0, bounds=bounds, maxfev=20000)
        y_fit = voigt_dip(x, *popt)
        return popt, self._r_squared(y, y_fit)

    def fit_double_voigt(self, x: np.ndarray, y: np.ndarray, guess_center: float):
        """Fits two Voigt dips at once. Initial guesses are found by locating
        the deepest point on either side of guess_center (left half / right
        half of the window), which matches the visual pattern of two
        overlapping dips straddling the catalog wavelength."""
        offset0 = float(np.max(y))
        span = x.max() - x.min()

        left_mask = x <= guess_center
        right_mask = x > guess_center

        if np.any(left_mask):
            cen1_0 = x[left_mask][np.argmin(y[left_mask])]
        else:
            cen1_0 = guess_center - span / 4

        if np.any(right_mask):
            cen2_0 = x[right_mask][np.argmin(y[right_mask])]
        else:
            cen2_0 = guess_center + span / 4

        amp0 = max(np.max(y) - np.min(y), 0.01) / 2
        sigma0 = max(span / 12.0, 0.5)
        gamma0 = max(span / 12.0, 0.5)

        p0 = [amp0, cen1_0, sigma0, gamma0, amp0, cen2_0, sigma0, gamma0, offset0]
        bounds = (
            [0, x.min(), 0.05, 0.0, 0, x.min(), 0.05, 0.0, 0.0],
            [10 * amp0 + 1e-6, guess_center, span, span,
             10 * amp0 + 1e-6, x.max(), span, span, 10 * offset0 + 1e-6],
        )
        # cen1 is constrained to [x.min(), guess_center] and cen2 to
        # [x.min(), x.max()] via bounds above -- but we also want cen2 to stay
        # on the right side, so tighten cen2's lower bound to guess_center.
        bounds[0][5] = guess_center

        popt, _ = curve_fit(voigt_double_dip, x, y, p0=p0, bounds=bounds, maxfev=40000)
        y_fit = voigt_double_dip(x, *popt)
        return popt, self._r_squared(y, y_fit)

    def fit_line(
        self,
        wavelength: np.ndarray,
        flux_norm: np.ndarray,
        line: SpectralLine,
        file_name: str,
    ) -> list[FitResult]:
        lo, hi = line.rest_wavelength - line.window, line.rest_wavelength + line.window
        mask = (wavelength >= lo) & (wavelength <= hi)
        x, y = wavelength[mask], flux_norm[mask]

        def _fail(note: str) -> list[FitResult]:
            """Always returns exactly line.n_components entries, so the
            flattened results list stays aligned with the catalog regardless
            of success/failure."""
            n = line.n_components
            comps = [""] if n == 1 else ["1", "2"]
            return [
                FitResult(
                    file_name=file_name,
                    line_name=line.name,
                    rest_wavelength=line.rest_wavelength,
                    fit_type="voigt_double" if n == 2 else "none",
                    component=comp,
                    note=note,
                )
                for comp in comps
            ]

        min_pts_needed = self.min_points * 2 if line.n_components == 2 else self.min_points
        if len(x) < min_pts_needed:
            return _fail("insufficient data points in window")

        # --- single-component Voigt fit (the default case) ---
        if line.n_components == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    popt, r2 = self.fit_voigt(x, y, line.rest_wavelength)
                except Exception as e:
                    return _fail(f"voigt fit failed to converge: {e}")

            amp, cen, sigma, gamma, offset = popt

            if self.verbose:
                print(f"    [{file_name}] {line.name}: voigt R^2={r2:.4f} "
                      f"(sigma={sigma:.2f}, gamma={gamma:.2f})")

            return [FitResult(
                file_name=file_name,
                line_name=line.name,
                rest_wavelength=line.rest_wavelength,
                fit_type="voigt",
                center=float(cen),
                depth=float(amp),
                width=float(sigma),
                extra_width=float(gamma),
                r_squared=float(r2),
                success=True,
                note="",
            )]

        # --- 2-component (blended/double-dip) Voigt fit ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt, r2 = self.fit_double_voigt(x, y, line.rest_wavelength)
            except Exception as e:
                return _fail(f"double voigt fit failed to converge: {e}")

        amp1, cen1, sigma1, gamma1, amp2, cen2, sigma2, gamma2, offset = popt

        if self.verbose:
            print(f"    [{file_name}] {line.name}: voigt_double R^2={r2:.4f} "
                  f"comp1(center={cen1:.1f}, sigma={sigma1:.2f}, gamma={gamma1:.2f}) "
                  f"comp2(center={cen2:.1f}, sigma={sigma2:.2f}, gamma={gamma2:.2f})")

        result1 = FitResult(
            file_name=file_name,
            line_name=line.name,
            rest_wavelength=line.rest_wavelength,
            fit_type="voigt_double",
            component="1",
            center=float(cen1),
            depth=float(amp1),
            width=float(sigma1),
            extra_width=float(gamma1),
            r_squared=float(r2),
            success=True,
            note="",
        )
        result2 = FitResult(
            file_name=file_name,
            line_name=line.name,
            rest_wavelength=line.rest_wavelength,
            fit_type="voigt_double",
            component="2",
            center=float(cen2),
            depth=float(amp2),
            width=float(sigma2),
            extra_width=float(gamma2),
            r_squared=float(r2),
            success=True,
            note="",
        )
        return [result1, result2]


# ---------------------------------------------------------------------------
# 6. SpectrumPlotter: saves a PNG showing the full spectrum + zoomed fit panels
# ---------------------------------------------------------------------------

class SpectrumPlotter:
    """Renders one PNG per spectrum: full normalized spectrum on top, and a
    zoomed-in panel per catalog line underneath showing the data + fitted curve."""

    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    @staticmethod
    def _group_results_by_line(
        line_catalog: list[SpectralLine], results: list[FitResult]
    ) -> list[list[FitResult]]:
        """Splits the flat results list back into one sub-list per catalog
        line, using each line's n_components (matches how fit_line/_fail
        always emit exactly n_components entries)."""
        grouped = []
        idx = 0
        for line in line_catalog:
            n = line.n_components
            grouped.append(results[idx: idx + n])
            idx += n
        return grouped

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

        results_by_line = self._group_results_by_line(line_catalog, results)

        fig = plt.figure(figsize=(14, 4 + 3 * n_rows))
        gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[2] + [1] * n_rows)

        # --- top panel: full normalized spectrum ---
        ax_full = fig.add_subplot(gs[0, :])
        ax_full.plot(wavelength, flux_norm, color="black", lw=0.6, label="spectrum")

        catalog_line_drawn = False
        fitted_line_drawn = False
        for line, line_results in zip(line_catalog, results_by_line):
            # Catalog (rest) wavelength -- where we searched.
            ax_full.axvline(
                line.rest_wavelength, color="tab:red", ls="--", lw=0.7, alpha=0.6,
                label="catalog wavelength" if not catalog_line_drawn else None,
            )
            catalog_line_drawn = True

            # Actual fitted center(s) -- where the Voigt fit(s) landed. Drawn
            # separately from the catalog line so a real offset (e.g. a
            # blended double-dip pulling away from the tabulated wavelength)
            # is visible at a glance in the overview panel, not just in the
            # zoomed panel below.
            for r in line_results:
                if r.success and np.isfinite(r.center):
                    ax_full.axvline(
                        r.center, color="tab:blue", ls="-", lw=0.9, alpha=0.85,
                        label="fitted center" if not fitted_line_drawn else None,
                    )
                    fitted_line_drawn = True

        ax_full.set_title(f"{file_name} \u2014 full normalized spectrum")
        ax_full.set_xlabel("Wavelength (\u00c5)")
        ax_full.set_ylabel("Normalized flux")
        ax_full.legend(fontsize=7, loc="lower right")

        # --- bottom panels: zoomed fit per line ---
        for i, (line, line_results) in enumerate(zip(line_catalog, results_by_line)):
            row, col = divmod(i, n_cols)
            ax = fig.add_subplot(gs[row + 1, col])

            lo, hi = line.rest_wavelength - line.window, line.rest_wavelength + line.window
            mask = (wavelength >= lo) & (wavelength <= hi)
            x, y = wavelength[mask], flux_norm[mask]
            ax.plot(x, y, "o", ms=3, color="black", label="data")

            all_success = len(line_results) > 0 and all(r.success for r in line_results)

            if all_success and len(line_results) == 1:
                result = line_results[0]
                x_fine = np.linspace(x.min(), x.max(), 200)
                y_fine = voigt_dip(x_fine, result.depth, result.center,
                                   result.width, result.extra_width, np.max(y))
                ax.plot(x_fine, y_fine, "-", color="tab:red", lw=1.5, label="voigt fit")
                ax.axvline(result.center, color="tab:blue", ls=":", lw=1)
                title = f"{line.name}\ncenter={result.center:.1f}\u00c5  R\u00b2={result.r_squared:.3f}"

            elif all_success and len(line_results) == 2:
                r1, r2 = line_results
                offset_guess = np.max(y)
                x_fine = np.linspace(x.min(), x.max(), 200)
                comp1 = voigt_dip(x_fine, r1.depth, r1.center, r1.width, r1.extra_width, offset_guess)
                comp2 = voigt_dip(x_fine, r2.depth, r2.center, r2.width, r2.extra_width, offset_guess)
                full = offset_guess - (offset_guess - comp1) - (offset_guess - comp2)
                ax.plot(x_fine, comp1, "--", color="tab:orange", lw=1.1, label="component 1")
                ax.plot(x_fine, comp2, "--", color="tab:green", lw=1.1, label="component 2")
                ax.plot(x_fine, full, "-", color="tab:red", lw=1.5, label="combined fit")
                ax.axvline(r1.center, color="tab:orange", ls=":", lw=1)
                ax.axvline(r2.center, color="tab:green", ls=":", lw=1)
                title = (f"{line.name} (2-comp)\n"
                         f"c1={r1.center:.1f}\u00c5  c2={r2.center:.1f}\u00c5  R\u00b2={r1.r_squared:.3f}")

            else:
                note = line_results[0].note if line_results else "no result"
                title = f"{line.name}\nfit failed: {note}"

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
            line_results = self.fitter.fit_line(wavelength, flux_norm, line, file_name)
            results.extend(line_results)

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