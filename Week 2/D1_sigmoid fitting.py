import os
from typing import Tuple, Sequence

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pymultifit.fitters.backend.baseFitter import BaseFitter
from astropy.io import fits


# ---------------------------------------------------------------------------
# Continuum model: sigmoid-blended pair of lines (user-provided)
# ---------------------------------------------------------------------------

def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def line(x, a, b):
    return a * x + b


def blended_model(x, transition, sharpness, a1, b1, a2, b2):
    """Continuum = sigmoid-weighted blend of two lines.

    Parameter order: (transition, sharpness, a1, b1, a2, b2)
      transition -> passed as sigmoid's k  (steepness)
      sharpness  -> passed as sigmoid's x0 (midpoint location)
      a1, b1     -> slope/intercept of line 1
      a2, b2     -> slope/intercept of line 2
    """
    s = sigmoid(x, transition, sharpness)
    return s * line(x, a1, b1) + (1 - s) * line(x, a2, b2)


class LinesWithSigmoid(BaseFitter):
    def __init__(self, x_values, y_values, max_iterations=1000):
        super().__init__(x_values, y_values, max_iterations)
        self.n_par = 6

    def fit_boundaries(self) -> Tuple[Sequence[float], Sequence[float]]:
        x_min, x_max = self.x_values.min(), self.x_values.max()
        y_min, y_max = self.y_values.min(), self.y_values.max()

        x_span = x_max - x_min
        y_span = y_max - y_min

        slope_bound = (y_span / x_span) * 10 if x_span > 0 else np.inf
        intercept_bound = 10 * max(abs(y_min), abs(y_max), 1.0)

        # Order MUST match blended_model's signature: (k, x0, a1, b1, a2, b2)
        lb = (-1.0, x_min, -slope_bound, -intercept_bound, -slope_bound, -intercept_bound)
        ub = (1.0, x_max, slope_bound, intercept_bound, slope_bound, intercept_bound)
        return lb, ub

    @staticmethod
    def fitter(x, params) -> np.ndarray:
        return blended_model(x, *params)


# ---------------------------------------------------------------------------
# Line-feature model: Gaussian only (positive amplitude for emission,
# negative amplitude for absorption)
# ---------------------------------------------------------------------------

def raw_gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


class GaussianFitter(BaseFitter):
    def __init__(self, x_values, y_values, max_iterations=1000):
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    def fit_boundaries(self):
        return (0, -np.inf, 0), (np.inf, np.inf, np.inf)

    @staticmethod
    def fitter(x, params):
        amplitude, mu, sigma = params
        return raw_gaussian(x, amplitude, mu, sigma)


class GaussianFitterNegativeAmplitude(GaussianFitter):
    def fit_boundaries(self):
        return (-np.inf, -np.inf, 0), (0, np.inf, np.inf)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_fits_spectrum(path, flux_hdu=0, table_hdu=None, wave_col="WAVELENGTH", flux_col="FLUX", wave_in_log10=False):
    with fits.open(path) as hdul:
        if table_hdu is not None:
            data = hdul[table_hdu].data
            wavelength = np.asarray(data[wave_col], dtype=float)
            flux = np.asarray(data[flux_col], dtype=float)
            if wave_in_log10:
                wavelength = 10 ** wavelength
            return wavelength, flux

        hdu = hdul[flux_hdu]
        flux = np.asarray(hdu.data, dtype=float).ravel()
        header = hdu.header
        crval1 = header["CRVAL1"]
        crpix1 = header.get("CRPIX1", 1)
        cdelt1 = header.get("CDELT1", header.get("CD1_1"))
        if cdelt1 is None:
            raise ValueError("No CDELT1/CD1_1 in header; try table_hdu=...")
        pixels = np.arange(1, len(flux) + 1)
        wavelength = crval1 + (pixels - crpix1) * cdelt1
        if wave_in_log10 or "LOG" in str(header.get("CTYPE1", "")).upper():
            wavelength = 10 ** wavelength
        return wavelength, flux


def load_sdss_redshift(path, specobj_hdu=2):
    with fits.open(path) as hdul:
        return float(hdul[specobj_hdu].data["Z"][0])


def expected_line_positions(z, known_lines=None):
    known_lines = known_lines or KNOWN_LINES
    return {name: rest * (1 + z) for name, rest in known_lines.items()}


KNOWN_LINES = {
    "H-alpha": 6562.8, "H-beta": 4861.3, "H-gamma": 4340.5, "H-delta": 4101.7,
    "Ca-K": 3933.7, "Ca-H": 3968.5, "Na-D": 5892.9,
}


# ---------------------------------------------------------------------------
# Continuum + feature fitting pipeline
# ---------------------------------------------------------------------------

def fit_continuum(wavelength, flux, p0=None):
    cont = LinesWithSigmoid(wavelength, flux)
    if p0 is None:
        # order: (transition/k, sharpness/x0, a1, b1, a2, b2)
        p0 = (0.001, float(np.median(wavelength)), 0.0, float(np.median(flux)), 0.0, float(np.median(flux)))
    cont.fit(p0=[p0])
    return cont


def detect_features(residual, prominence_sigma=4.0):
    noise = np.std(residual)
    peaks, _ = find_peaks(residual, prominence=prominence_sigma * noise)
    dips, _ = find_peaks(-residual, prominence=prominence_sigma * noise)
    return peaks, dips


def fit_local_feature(wavelength, residual, idx, window=20, is_peak=True):
    lo, hi = max(0, idx - window), min(len(wavelength), idx + window)
    xw, yw = wavelength[lo:hi], residual[lo:hi]
    amp0, mu0 = residual[idx], wavelength[idx]
    width0 = max((xw[-1] - xw[0]) / 6.0, 1e-3)

    FitterClass = GaussianFitter if is_peak else GaussianFitterNegativeAmplitude
    f = FitterClass(xw, yw)
    try:
        f.fit(p0=[(amp0, mu0, width0)])
        amplitude, mu, sigma = f.get_model_parameters()[:, 0]
    except (RuntimeError, ValueError):
        return None

    return {"mu": mu, "amplitude": amplitude, "width": sigma, "fwhm": 2.3548 * sigma, "profile": "gaussian"}


def match_lines(fitted_centers, known_lines=KNOWN_LINES, max_shift_fraction=0.02, steps=4001):
    known_wl = np.array(list(known_lines.values()))
    known_names = list(known_lines.keys())
    best = None
    for z in np.linspace(-max_shift_fraction, max_shift_fraction, steps):
        shifted = known_wl * (1 + z)
        matches, total_err = [], 0.0
        for mu in fitted_centers:
            j = int(np.argmin(np.abs(shifted - mu)))
            err = abs(shifted[j] - mu)
            total_err += err
            matches.append((known_names[j], shifted[j], mu, err))
        if best is None or total_err < best[0]:
            best = (total_err, z, matches)
    return best


def analyze_spectrum(wavelength, flux, mask=None, prominence_sigma=4.0, window=20, max_match_error=5.0):
    wavelength, flux = np.asarray(wavelength), np.asarray(flux)
    if mask is not None:
        cont_fit = fit_continuum(wavelength[~mask], flux[~mask])
    else:
        cont_fit = fit_continuum(wavelength, flux)
    params = np.ravel(cont_fit.get_model_parameters())
    continuum_full = blended_model(wavelength, *params)
    residual = flux - continuum_full
    peaks, dips = detect_features(residual, prominence_sigma)

    results = []
    for idx in peaks:
        r = fit_local_feature(wavelength, residual, idx, window, is_peak=True)
        if r:
            r["type"] = "emission"
            results.append(r)
    for idx in dips:
        r = fit_local_feature(wavelength, residual, idx, window, is_peak=False)
        if r:
            r["type"] = "absorption"
            results.append(r)

    shift_info = None
    if results:
        centers = [r["mu"] for r in results]
        total_err, z, matches = match_lines(centers)
        for r, (name, rest_shifted, mu, err) in zip(results, matches):
            if err <= max_match_error:
                r["identified_as"] = name
                r["rest_wavelength"] = KNOWN_LINES[name]
                r["match_error"] = err
            else:
                r["identified_as"] = None
                r["match_error"] = err
        shift_info = {"shift_fraction": z, "total_match_error": total_err}

    return {"continuum_params": params, "continuum_curve": continuum_full, "residual": residual,
            "lines": results, "shift_info": shift_info}


FITS_PATH = r"C:\Users\Ayank\OneDrive\Desktop\internship\Week 2\spec-0417-51821-0428.fits"


def make_synthetic_spectrum():
    rng = np.random.default_rng(0)
    wavelength = np.linspace(3700, 8000, 2000)
    # continuum params in blended_model order: (k, x0, a1, b1, a2, b2)
    continuum_true = blended_model(wavelength, 0.0013, 5200, -0.00045, 6.7, 0.033, 40.0)
    z_true = 0.003
    injected = {
        "H-alpha": (-4.0, KNOWN_LINES["H-alpha"] * (1 + z_true), 6.0),
        "H-beta": (-3.0, KNOWN_LINES["H-beta"] * (1 + z_true), 5.0),
        "H-gamma": (-2.2, KNOWN_LINES["H-gamma"] * (1 + z_true), 4.5),
        "Na-D": (2.5, KNOWN_LINES["Na-D"] * (1 + z_true), 5.0),
    }
    flux = continuum_true.copy()
    for amp, mu, sigma in injected.values():
        flux += raw_gaussian(wavelength, amp, mu, sigma)
    flux += rng.normal(0, 0.15, size=wavelength.size)
    return wavelength, flux


if __name__ == "__main__":
    if os.path.exists(FITS_PATH):
        wavelength, flux = load_fits_spectrum(FITS_PATH, table_hdu=1, wave_col="loglam", flux_col="flux", wave_in_log10=True)
    else:
        print(f"'{FITS_PATH}' not found -- using synthetic demo data.\n")
        wavelength, flux = make_synthetic_spectrum()

    result = analyze_spectrum(wavelength, flux, prominence_sigma=4.0, window=25, max_match_error=5.0)

    sdss_z = None
    if os.path.exists(FITS_PATH):
        try:
            sdss_z = load_sdss_redshift(FITS_PATH)
            print(f"SDSS pipeline redshift (SPECOBJ.Z): {sdss_z:.5f}")
        except Exception as e:
            print(f"Could not read SDSS redshift: {e}")

    print(f"Continuum params (k, x0, a1, b1, a2, b2): {result['continuum_params']}")
    if result["shift_info"]:
        print(f"Shift fraction from detected lines: {result['shift_info']['shift_fraction']:.5f}")
    for r in result["lines"]:
        print(f"{r['type']:10s} {str(r.get('identified_as')):10s} profile={r['profile']:10s} mu={r['mu']:.2f} "
              f"match_err={r.get('match_error', float('nan')):.2f} amp={r['amplitude']:.2f} fwhm={r['fwhm']:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].scatter(wavelength, flux, s=3, color="gray", alpha=0.4, label="data")
    axes[0].plot(wavelength, result["continuum_curve"], color="crimson", lw=2, label="continuum")
    axes[0].legend()
    axes[1].plot(wavelength, result["residual"], color="gray", lw=1, label="residual")

    if sdss_z is not None:
        for name, expected_mu in expected_line_positions(sdss_z).items():
            if wavelength.min() <= expected_mu <= wavelength.max():
                axes[1].axvline(expected_mu, color="green", ls=":", alpha=0.5)
                axes[1].annotate(name, (expected_mu, axes[1].get_ylim()[1] * 0.9),
                                  fontsize=7, color="green", ha="center")

    for r in result["lines"]:
        if r.get("identified_as") is None:
            continue
        color = "royalblue" if r["type"] == "absorption" else "darkorange"
        axes[1].axvline(r["mu"], color=color, ls="--", alpha=0.7)
        axes[1].annotate(r["identified_as"], (r["mu"], r["amplitude"]),
                          textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].legend()
    plt.tight_layout()
    plt.show()