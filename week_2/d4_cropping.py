import os
import glob

from src.internship.utilities import crop_and_check

INPUT_FOLDER = r"C:\Users\Ayank\OneDrive\Desktop\internship\spectra_files"
OUTPUT_DIR = os.path.join(INPUT_FOLDER, "processed_spectrum")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.fits"))

    if not files:
        print(f"No .fits files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(files)} FITS file(s) in {INPUT_FOLDER}")

    for f in files:
        print(f"\nProcessing {f} ...")
        crop_and_check(f, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()