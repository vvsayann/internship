import sys
print(sys.executable)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz


def voigt_pdf(x, loc, sigma, gamma):
    z = ((x - loc) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def voigt(x, amplitude, mu, sigma, gamma):
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z))
    peak = np.real(wofz(1j * gamma / (sigma * np.sqrt(2))))
    return amplitude * profile / peak


params = [
    (0, 1, 1),
    (-3, 2, 3),
    (0.5, 1.5, 2),
]
x = np.linspace(-20, 20, 2000)

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)

for (mu, sigma, gamma), ax in zip(params, axes):
    pdf = voigt_pdf(x, mu, sigma, gamma)
    norm_pdf = pdf / np.max(pdf)
    amp_calc = np.max(pdf)
    main = voigt(x, amp_calc, mu, sigma, gamma)

    ax.plot(x, pdf, label="voigt_pdf (area=1)", lw=2)
    ax.plot(x, main, "-", label=f"voigt (amp={amp_calc:.3f})", lw=2)
    ax.plot(x, norm_pdf, label="normalised (peak=1)", lw=2)

    ax.set_title(f"mu={mu}, sigma={sigma}, gamma={gamma}")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("value")
plt.tight_layout()
plt.savefig("voigts.png", dpi=150)
plt.show()