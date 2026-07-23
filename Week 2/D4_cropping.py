
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ------------------------- CONFIG -------------------------
# If you don't pass filenames on the command line, list them here instead:
FILES_TO_CHECK = [
    "spec-0266-51630-0098.fits",
]

HDU_INDEX = 1              # SDSS spectra store the table in extension 1
FLUX_COLUMN = "flux"
LOGLAM_COLUMN = "loglam"

MAX_WAVELENGTH = 7500.0    # angstroms — crop cutoff

OUTPUT_DIR = r"C:\Users\Ayank\OneDrive\Desktop\Spectra 2"  # where cropped FITS + plots go
# ------------------------------------------------------------


def crop_and_check(filepath):
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

        # ---- Save cropped FITS, preserving all other HDUs ----
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_fits = os.path.join(OUTPUT_DIR, f"{base}_cropped7500.fits")

        new_hdu = fits.BinTableHDU(data=trimmed_data, header=header)
        hdul_out = fits.HDUList()
        for i, hdu in enumerate(hdul):
            hdul_out.append(new_hdu if i == HDU_INDEX else hdu.copy())
        hdul_out.writeto(out_fits, overwrite=True)
        print(f"  Saved cropped FITS: {out_fits}")

        # ---- Save a check plot ----
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
        out_png = os.path.join(OUTPUT_DIR, f"{base}_cropped7500_check.png")
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  Saved check plot: {out_png}")


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else FILES_TO_CHECK

    if not files:
        print("No files specified. Pass filenames as arguments or edit FILES_TO_CHECK.")
        return

    for f in files:
        crop_and_check(f)

    print("\nDone.")


if __name__ == "__main__":
    main()