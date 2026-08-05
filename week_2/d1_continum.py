from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

from src.internship.fitters import LinesWithSigmoid
from src.internship.utilities import blended_model


def load_sdss_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    with fits.open(filepath) as hdul:
        data = hdul[1].data
        wavelength = 10 ** data["loglam"]
        flux = data["flux"]
        ivar = data["ivar"]

    good = ivar > 0
    return wavelength[good], flux[good]


FITS_PATH = r"C:\Users\Ayank\OneDrive\Desktop\internship\spectra_files\spec-0417-51821-0428.fits"

x, y_data = load_sdss_spectrum(FITS_PATH)

y_mean = np.median(y_data)
x_mid = 0.5 * (x.min() + x.max())
x_span = x.max() - x.min()
p0 = [0.0, y_mean, 0.0, y_mean, 4.0 / x_span, x_mid]

fitter_obj = LinesWithSigmoid(x, y_data)
lb, ub = fitter_obj.fit_boundaries()

params, cov = curve_fit(blended_model, x, y_data, p0=p0, bounds=(lb, ub), maxfev=20000)
a1, b1, a2, b2, k, x0 = params

print("Fitted parameters:")
print(f"  L1: y = {a1:.6f}*x + {b1:.3f}")
print(f"  L2: y = {a2:.6f}*x + {b2:.3f}")
print(f"  sigmoid: k = {k:.6f}, x0 = {x0:.1f}")

final = blended_model(x, a1, b1, a2, b2, k, x0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y_data, s=5, alpha=0.3, color="gray", label="data")
ax.plot(x, final, "k", lw=2, label="Final fit")
ax.set_xlabel("Wavelength")
ax.set_ylabel("Flux")
ax.set_title("Spectrum with blended-line fit")
ax.legend()
plt.tight_layout()
plt.show()
