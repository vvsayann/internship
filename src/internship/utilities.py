from __future__ import annotations

import os
import sys

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.special._ufuncs import wofz

KNOWN_LINES = {
    "H-alpha": 6562.8, "H-beta": 4861.3, "H-gamma": 4340.5, "H-delta": 4101.7,
    "Ca-K": 3933.7, "Ca-H": 3968.5, "Na-D": 5892.9,
}


def lorentzian_dip(x, amp, cen, gamma, offset):
    gamma = max(gamma, 1e-6)
    return offset - amp * (gamma ** 2) / ((x - cen) ** 2 + gamma ** 2)


def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def gaussian_dip(x, amp, cen, sigma, offset):
    sigma = max(sigma, 1e-6)
    return offset - amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))


def raw_gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sigmoid_dip(x, amp, cen, width, offset):
    return offset - amp / (1 + np.exp((x - cen) / width))


def vacumm_to_air(wavelength: np.ndarray) -> np.ndarray:
    term1 = 5.792105E-2 / (238.0185 - (1.0E4 / wavelength) ** 2)
    term2 = 1.67917E-3 / (57.362 - (1.0E4 / wavelength) ** 2)
    air = wavelength / (1.0 + term1 + term2)
    return air


FILES_TO_CHECK = [
    "file name",
]

HDU_INDEX = 1
FLUX_COLUMN = "flux"
LOGLAM_COLUMN = "loglam"

MAX_WAVELENGTH = 7500.0

OUTPUT_DIR = r"your output directory"


def crop_and_check(filepath, output_dir):
    print(f"\n--- Checking {filepath} ---")

    if not os.path.exists(filepath):
        print(f"  SKIPPED: file not found")
        return

    with fits.open(filepath) as hdul:
        data = hdul[HDU_INDEX].data.copy()
        header = hdul[HDU_INDEX].header

        loglam = np.array(data[LOGLAM_COLUMN], dtype=float)
        wavelength = 10 ** loglam
        flux = np.array(data[FLUX_COLUMN], dtype=float)

        print(f"  Original range: {wavelength.min():.2f} - {wavelength.max():.2f} A "
              f"({len(wavelength)} points)")

        mask = wavelength <= MAX_WAVELENGTH
        n_kept = mask.sum()

        if n_kept == 0:
            print(f"  WARNING: no points below {MAX_WAVELENGTH} A — spectrum starts "
                  f"above the cutoff. Nothing to save.")
            return

        trimmed_data = data[mask]
        trimmed_wavelength = wavelength[mask]
        trimmed_flux = flux[mask]

        print(f"  Cropped range: {trimmed_wavelength.min():.2f} - "
              f"{trimmed_wavelength.max():.2f} A ({n_kept} points kept, "
              f"{len(wavelength) - n_kept} discarded)")

        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_fits = os.path.join(output_dir, f"{base}.fits")
        new_hdu = fits.BinTableHDU(data=trimmed_data, header=header)
        hdul_out = fits.HDUList()
        for i, hdu in enumerate(hdul):
            hdul_out.append(new_hdu if i == HDU_INDEX else hdu.copy())
        hdul_out.writeto(out_fits, overwrite=True)
        print(f"  Saved cropped FITS: {out_fits}")

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))

        axes[0].plot(wavelength, flux, color="red", linewidth=0.7)
        axes[0].axvline(MAX_WAVELENGTH, color="blue", linestyle="--", label="cutoff (7500 A)")
        axes[0].set_title(f"Original: {filepath}")
        axes[0].set_xlabel("Wavelength (A)")
        axes[0].legend()

        axes[1].plot(trimmed_wavelength, trimmed_flux, color="green", linewidth=0.7)
        axes[1].set_title(f"Cropped to <= {MAX_WAVELENGTH} A")
        axes[1].set_xlabel("Wavelength (A)")

        plt.tight_layout()
        out_png = os.path.join(output_dir, f"{base}.png")
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  Saved check plot: {out_png}")


INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "spectrum.fits"
OUTPUT_FILE = INPUT_FILE.replace(".fits", "_despiked.fits")

HDU_INDEX = 1
IS_TABLE = True
FLUX_COLUMN = "FLUX"

WINDOW = 15
GAP = 3
SIGMA_THRESH = 3.5
PASSES = 3


def detect_spikes(data, window=WINDOW, sigma_thresh=SIGMA_THRESH, gap=GAP):
    n = len(data)
    spikes = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        buf_lo, buf_hi = max(lo, i - gap), min(hi, i + gap + 1)
        local = np.concatenate([data[lo:buf_lo], data[buf_hi:hi]])
        if local.size < 4:
            continue
        med = np.median(local)
        robust_std = mad_std(local)
        if robust_std > 0 and abs(data[i] - med) > sigma_thresh * robust_std:
            spikes[i] = True
    return spikes


def remove_spikes(data, spikes):
    clean = data.copy()
    idx = np.arange(len(data))
    if spikes.any():
        clean[spikes] = np.interp(idx[spikes], idx[~spikes], data[~spikes])
    return clean


def detect_and_remove_spikes(data, window=WINDOW, sigma_thresh=SIGMA_THRESH,
                             gap=GAP, passes=PASSES):
    all_spikes = np.zeros(len(data), dtype=bool)
    clean = data.copy()
    for _ in range(passes):
        new_spikes = detect_spikes(clean, window, sigma_thresh, gap) & ~all_spikes
        if not new_spikes.any():
            break
        all_spikes |= new_spikes
        clean = remove_spikes(data, all_spikes)
    return clean, all_spikes


def blended_model(x, a1, b1, a2, b2, k, x0):
    s = sigmoid(x, k, x0)
    return s * line(x, a1, b1) + (1 - s) * line(x, a2, b2)


def line(x, a, b):
    return a * x + b


def _blended_model(x, transition, sharpness, a1, b1, a2, b2):
    s = sigmoid(x, transition, sharpness)
    return s * line(x, a1, b1) + (1 - s) * line(x, a2, b2)


def detect_features(residual, prominence_sigma=4.0):
    noise = np.std(residual)
    peaks, _ = find_peaks(residual, prominence=prominence_sigma * noise)
    dips, _ = find_peaks(-residual, prominence=prominence_sigma * noise)
    return peaks, dips


def voigt_dip(x, amp, cen, sigma, gamma, offset):
    return offset - amp * _voigt_peak_normalized(x, cen, sigma, gamma)


def _voigt_peak_normalized(x, cen, sigma, gamma):
    sigma = max(sigma, 1e-6)
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z))
    peak = np.real(wofz(1j * gamma / (sigma * np.sqrt(2))))
    peak = peak if peak > 1e-12 else 1e-12
    return profile / peak


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
