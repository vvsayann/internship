from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpectralLine:
    name: str
    rest_wavelength: float
    window: float = 15.0