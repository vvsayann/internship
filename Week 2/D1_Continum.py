from typing import Tuple, Sequence

import numpy as np
from astropy.io import fits
from pymultifit.fitters.backend import BaseFitter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def load_sdss_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:

    with fits.open(filepath) as hdul:
        data = hdul[1].data
        wavelength = 10 ** data["loglam"]
        flux = data["flux"]
        ivar = data["ivar"]


    good = ivar > 0
    return wavelength[good], flux[good]


def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def line(x, a, b):
    return a * x + b


def blended_model(x, a1, b1, a2, b2, k, x0):

    s = sigmoid(x, k, x0)
    return s * line(x, a1, b1) + (1 - s) * line(x, a2, b2)


class LinesWithSigmoid(BaseFitter):

    def __init__(self, x_values, y_values, max_iterations=1000):
        super().__init__(x_values, y_values, max_iterations)
        self.n_par = 6
        self.n_fits = 1

    def fit_boundaries(self) -> Tuple[Sequence[float], Sequence[float]]:
        x_min, x_max = self.x_values.min(), self.x_values.max()
        y_min, y_max = self.y_values.min(), self.y_values.max()

        x_span = x_max - x_min
        y_span = y_max - y_min

        slope_bound = (y_span / x_span) * 10 if x_span > 0 else np.inf
        intercept_bound = 10 * max(abs(y_min), abs(y_max), 1.0)


        lb = (-slope_bound, -intercept_bound, -slope_bound, -intercept_bound, -1.0, x_min)
        ub = (slope_bound, intercept_bound, slope_bound, intercept_bound, 1.0, x_max)
        return lb, ub

    @staticmethod
    def fitter(x, params) -> np.ndarray:
        return blended_model(x, *params)



FITS_PATH = r"C:\Users\Ayank\OneDrive\Desktop\internship\Week 2\spec-0266-51630-0098.fits"
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

L1 = line(x, a1, b1)
L2 = line(x, a2, b2)
s = sigmoid(x, k, x0)
contrib_L1 = s * L1
contrib_L2 = (1 - s) * L2
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
