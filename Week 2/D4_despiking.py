import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import mad_std

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "spectrum.fits"
OUTPUT_FILE = INPUT_FILE.replace(".fits", "_despiked.fits")

HDU_INDEX = 1
IS_TABLE = True
FLUX_COLUMN = "FLUX"

WINDOW = 15
GAP = 3
SIGMA_THRESH = 3.5
PASSES = 3


def load_spectrum(path):
    with fits.open(path) as hdul:
        header = hdul[HDU_INDEX].header.copy()
        if IS_TABLE:
            data = np.array(hdul[HDU_INDEX].data[FLUX_COLUMN], dtype=float)
        else:
            data = np.array(hdul[HDU_INDEX].data, dtype=float)
    return data, header


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