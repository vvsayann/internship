import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from src.internship.utilities import detect_and_remove_spikes

INPUT_FOLDER = r"C:\Users\Ayank\OneDrive\Desktop\internship\spectra_files"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "despiked_spectrum")

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


def save_result(input_path, output_path, data, header):
    if IS_TABLE:
        with fits.open(input_path) as hdul:
            hdul[HDU_INDEX].data[FLUX_COLUMN] = data
            hdul.writeto(output_path, overwrite=True)
    else:
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_path, overwrite=True)


def plot_result(original, clean, spikes, plot_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(original, color="red", linewidth=0.7)
    axes[0].scatter(np.where(spikes)[0], original[spikes],
                     color="black", marker="x", s=60, label="detected spike")
    axes[0].set_title("Original spectrum (spikes marked)")
    axes[0].legend()

    axes[1].plot(clean, color="green", linewidth=0.7)
    axes[1].set_title("Despiked spectrum")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {plot_path}")


def process_file(input_path, output_folder):
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, filename.replace(".fits", "_despiked.fits"))
    plot_path = os.path.join(output_folder, filename.replace(".fits", "_before_after.png"))

    print(f"\nReading {input_path} ...")
    data, header = load_spectrum(input_path)

    print("Detecting and removing spikes ...")
    clean, spikes = detect_and_remove_spikes(data)
    print(f"Found {spikes.sum()} spike point(s) at indices: {np.where(spikes)[0]}")

    print(f"Writing cleaned spectrum to {output_path} ...")
    save_result(input_path, output_path, clean, header)

    plot_result(data, clean, spikes, plot_path)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    fits_files = glob.glob(os.path.join(INPUT_FOLDER, "*.fits"))

    if not fits_files:
        print(f"No .fits files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(fits_files)} FITS file(s) in {INPUT_FOLDER}")

    for input_path in fits_files:
        process_file(input_path, OUTPUT_FOLDER)

    print("\nDone.")


if __name__ == "__main__":
    main()