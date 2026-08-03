import sys

from src.internship.utilities import crop_and_check

FILES_TO_CHECK = [
    "spec-0266-51630-0098.fits",
]

HDU_INDEX = 1
FLUX_COLUMN = "flux"
LOGLAM_COLUMN = "loglam"

MAX_WAVELENGTH = 7500.0

OUTPUT_DIR = r"your output directory"


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