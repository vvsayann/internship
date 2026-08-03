from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdul = fits.open('spec-0289-51990-0204.fits')

header = hdul[0].header
data = hdul['COADD'].data

flux_raw = data['flux']
loglam = data['loglam']
z = hdul[2].data.Z[0]

def vacumm_to_air(wavelength: np.ndarray) -> np.ndarray:
    term1 = 5.792105E-2 / (238.0185 - (1.0E4 / wavelength)**2)
    term2 = 1.67917E-3 / (57.362 - (1.0E4 / wavelength)**2)
    air = wavelength / (1.0 + term1 + term2)
    return air

wavelength_angstrom = 10**loglam / (1 + z)
wavelength_m = vacumm_to_air(wavelength_angstrom)

flux = flux_raw * 1e-17

print("Wavelength (A):", wavelength_m)
print("Flux (erg/s/cm^2/A):", flux)

plt.figure(figsize=(15,10))
plt.plot(wavelength_m, flux, color='red', lw=0.5)

# Spectral line wavelengths (Angstrom)
spectral_lines = {
    "He II 4686": 4686.0,
    "He I 4471": 4471.0,
    "Hα 6562.8": 6562.8,
    "Hβ 4861.3": 4861.3,
    "Hγ 4340.5": 4340.5
}


for label, wavelength in spectral_lines.items():
    plt.axvline(x=wavelength, color='blue', linestyle='-', linewidth=1)
    plt.text(
        wavelength,
        max(flux) * 0.95,
        label,
        rotation=90,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=9,
        color='blue'
    )

plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Flux (erg/s/cm²/Å)')
plt.title('Flux vs Wavelength')
plt.grid(alpha=0.5)

plt.show()

hdul.close()