from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FitResult:
    file_name: str
    line_name: str
    rest_wavelength: float
    fit_type: str = "none"
    center: float = np.nan
    depth: float = np.nan
    width: float = np.nan
    gamma: float = np.nan
    r_squared: float = np.nan
    success: bool = False
    note: str = ""
    continuum_offset: float = np.nan

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "line_name": self.line_name,
            "rest_wavelength": self.rest_wavelength,
            "fit_type": self.fit_type,
            "fitted_center": self.center,
            "depth": self.depth,
            "sigma_gaussian_width": self.width,
            "gamma_voigt": self.gamma,
            "r_squared": self.r_squared,
            "success": self.success,
            "note": self.note,
        }