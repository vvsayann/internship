"""
despike_fits.py

Removes single/narrow-point spike artifacts (e.g. cosmic ray hits, and
narrow sky-emission-line subtraction residuals such as the 5577/6300/6363 A
night-sky lines that are extremely common in SDSS-style spectra) from a 1D
FITS spectrum, saves the cleaned spectrum to a NEW file (original is never
modified), and shows a before/after plot so you can verify the fix.

Usage:
    python despike_fits.py input_spectrum.fits

Adjust the CONFIG section below to match your file's structure.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import mad_std

# ------------------------- CONFIG -------------------------
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "spectrum.fits"
OUTPUT_FILE = INPUT_FILE.replace(".fits", "_despiked.fits")

HDU_INDEX = 1          # which HDU holds the data (0 = primary; try 1 if that's empty)
IS_TABLE = True         # set True if data lives in a table column instead of an image array
FLUX_COLUMN = "FLUX"    # only used if IS_TABLE is True

WINDOW = 15             # number of neighboring points on EACH side used for local stats
GAP = 3                 # points immediately adjacent to the tested point that are EXCLUDED
                        # from the local baseline. This is what protects against multi-pixel
                        # wide artifacts (cosmic rays hitting at an angle, sky-line residuals)
                        # contaminating the very statistics used to judge them.
SIGMA_THRESH = 3.5      # how many local-sigma above/below median counts as a spike
PASSES = 3              # re-run detection on the progressively cleaned data up to this many
                        # times, so a point initially masked by a bigger neighboring spike
                        # still gets caught once that neighbor has been smoothed away
# ------------------------------------------------------------


def load_spectrum(path):
    with fits.open(path) as hdul:
        header = hdul[HDU_INDEX].header.copy()
        if IS_TABLE:
            data = np.array(hdul[HDU_INDEX].data[FLUX_COLUMN], dtype=float)
        else:
            data = np.array(hdul[HDU_INDEX].data, dtype=float)
    return data, header


def detect_spikes(data, window=WINDOW, sigma_thresh=SIGMA_THRESH, gap=GAP):
    """Flag points that deviate strongly from their local neighborhood.

    Two changes versus a naive version matter for artifacts wider than a
    single pixel:

    1. `gap` excludes a small buffer around the tested point (not just the
       point itself) from the reference window. Without this, the OTHER
       pixels belonging to the same artifact stay inside the "local"
       sample: they drag the local median around and inflate the local
       spread so much that even the most extreme point in the group no
       longer looks like an outlier relative to it. Nearby outliers end up
       masking each other - this is why a single-point exclusion misses
       multi-pixel spikes even at a very obvious visual amplitude.
    2. The local spread is estimated with the median absolute deviation
       (scaled to be comparable to a standard deviation) instead of a
       plain std. MAD is far less sensitive to a handful of contaminating
       points than std is, so it stays informative even when a few of the
       remaining "local" points are still a bit off.
    """
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
    """Runs detection/removal for up to `passes` rounds. Each round looks for
    NEW spikes in the best-current cleaned version of the data (so a smaller
    point that was masked by a bigger neighboring spike in round 1 becomes
    visible once that neighbor has been smoothed away), but always
    interpolates from the ORIGINAL data at not-yet-flagged points, so
    replacement values never compound across rounds.
    """
    all_spikes = np.zeros(len(data), dtype=bool)
    clean = data.copy()
    for _ in range(passes):
        new_spikes = detect_spikes(clean, window, sigma_thresh, gap) & ~all_spikes
        if not new_spikes.any():
            break
        all_spikes |= new_spikes
        clean = remove_spikes(data, all_spikes)
    return clean, all_spikes


def save_result(path, data, header):
    if IS_TABLE:
        with fits.open(INPUT_FILE) as hdul:
            hdul[HDU_INDEX].data[FLUX_COLUMN] = data
            hdul.writeto(path, overwrite=True)
    else:
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(path, overwrite=True)


def plot_result(original, clean, spikes):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(original, color="red", linewidth=0.7)
    axes[0].scatter(np.where(spikes)[0], original[spikes],
                     color="black", marker="x", s=60, label="detected spike")
    axes[0].set_title("Original spectrum (spikes marked)")
    axes[0].legend()

    axes[1].plot(clean, color="green", linewidth=0.7)
    axes[1].set_title("Despiked spectrum")

    plt.tight_layout()
    plt.savefig("despike_before_after.png", dpi=150)
    print("Saved comparison plot to despike_before_after.png")


def main():
    print(f"Reading {INPUT_FILE} ...")
    data, header = load_spectrum(INPUT_FILE)

    print("Detecting and removing spikes ...")
    clean, spikes = detect_and_remove_spikes(data)
    print(f"Found {spikes.sum()} spike point(s) at indices: {np.where(spikes)[0]}")

    print(f"Writing cleaned spectrum to {OUTPUT_FILE} (original file untouched) ...")
    save_result(OUTPUT_FILE, clean, header)

    plot_result(data, clean, spikes)
    print("Done.")


if __name__ == "__main__":
    main()