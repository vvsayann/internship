import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from src.internship.utilities import detect_and_remove_spikes

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