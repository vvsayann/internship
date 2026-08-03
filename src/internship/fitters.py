from __future__ import annotations

import warnings
from typing import Tuple, Sequence

import numpy as np
from pymultifit.fitters.backend import BaseFitter
from scipy.optimize import curve_fit

from .fit_result import FitResult
from .spectrum import SpectralLine
from .utilities import gaussian_dip, raw_gaussian, sigmoid_dip, blended_model


class GaussianFitter(BaseFitter):
    def __init__(self, x_values, y_values, max_iterations=1000):
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    def fit_boundaries(self):
        return (0, -np.inf, 0), (np.inf, np.inf, np.inf)

    @staticmethod
    def fitter(x, params):
        amplitude, mu, sigma = params
        return raw_gaussian(x, amplitude, mu, sigma)


class LineFitter:
    def __init__(self, min_points: int = 8):
        self.min_points = min_points

    @staticmethod
    def _r_squared(y_obs: np.ndarray, y_fit: np.ndarray) -> float:
        ss_res = np.sum((y_obs - y_fit) ** 2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def fit_gaussian(self, x: np.ndarray, y: np.ndarray, guess_center: float):
        amp0 = max(np.max(y) - np.min(y), 0.01)
        sigma0 = max((x.max() - x.min()) / 6.0, 0.5)
        offset0 = float(np.max(y))
        p0 = [amp0, guess_center, sigma0, offset0]
        bounds = (
            [0, x.min(), 0.1, 0.0],
            [10 * amp0 + 1e-6, x.max(), (x.max() - x.min()), 10 * offset0 + 1e-6],
        )
        popt, _ = curve_fit(gaussian_dip, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return popt, self._r_squared(y, gaussian_dip(x, *popt))

    def fit_sigmoid(self, x: np.ndarray, y: np.ndarray, guess_center: float):
        amp0 = max(np.max(y) - np.min(y), 0.01)
        width0 = max((x.max() - x.min()) / 8.0, 0.5)
        offset0 = float(np.max(y))
        p0 = [amp0, guess_center, width0, offset0]
        bounds = (
            [0, x.min(), 0.05, 0.0],
            [10 * amp0 + 1e-6, x.max(), (x.max() - x.min()), 10 * offset0 + 1e-6],
        )
        popt, _ = curve_fit(sigmoid_dip, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return popt, self._r_squared(y, sigmoid_dip(x, *popt))

    def fit_line(self, wavelength: np.ndarray, flux_norm: np.ndarray, line: SpectralLine, file_name: str) -> FitResult:
        lo, hi = line.rest_wavelength - line.window, line.rest_wavelength + line.window
        mask = (wavelength >= lo) & (wavelength <= hi)
        x, y = wavelength[mask], flux_norm[mask]

        if len(x) < self.min_points:
            return FitResult(
                file_name=file_name,
                line_name=line.name,
                rest_wavelength=line.rest_wavelength,
                note="insufficient data points in window",
            )

        candidates = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                popt, r2 = self.fit_gaussian(x, y, line.rest_wavelength)
                candidates.append(("gaussian", popt, r2))
            except Exception:
                pass

            try:
                popt, r2 = self.fit_sigmoid(x, y, line.rest_wavelength)
                candidates.append(("sigmoid", popt, r2))
            except Exception:
                pass

        if not candidates:
            return FitResult(
                file_name=file_name,
                line_name=line.name,
                rest_wavelength=line.rest_wavelength,
                note="both fits failed to converge",
            )

        fit_type, popt, r2 = max(candidates, key=lambda r: r[2])
        amp, cen, width_param, offset = popt

        return FitResult(
            file_name=file_name,
            line_name=line.name,
            rest_wavelength=line.rest_wavelength,
            fit_type=fit_type,
            center=float(cen),
            depth=float(amp),
            width=float(width_param),
            r_squared=float(r2),
            success=True,
        )


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
