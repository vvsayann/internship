import numpy as np
import matplotlib.pyplot as plt


def lorentzian_pdf(x, loc, scale):
    """
    x : array_like
        points at which we will calculate.
    loc : float
        parameter of the location.
    scale : float
        defines the parameter of the scale.

    Returns array_like
        Lorentzian distribution evaluated at a point x.
    """
    z = (x - loc) / scale
    return 1 / (np.pi * scale * (1 + z ** 2))


def lorentzian(x, amplitude, mu, gamma):
    """
    x : array_like
        Input wavelength (or frequency) values.
    amplitude : float
        maximum height of the distribution.
    mu : float
        Wavelength at which the maximum height is located.
    gamma : float
        Determines how spread out the distribution is.

    Returns array_like
        Lorentzian profile evaluated at x.
    """
    return amplitude * gamma ** 2 / ((x - mu) ** 2 + gamma ** 2)



params = [
    (0, 1),
    (-3, 3),
    (0.5, 2),
]
amplitude = 3 
x = np.linspace(-20, 20, 2000)   

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)

for (mu, gamma), ax in zip(params, axes):
    pdf = lorentzian_pdf(x, mu, gamma)
    norm_pdf = pdf / np.max(pdf)                 
    amplitude = 1 / (np.pi * gamma)               
    main = lorentzian(x, amplitude, mu, gamma)     

    ax.plot(x, pdf, label="lorentzian_pdf (area=1)", lw=2)
    ax.plot(x, main, "-", label=f"lorentzian (amp={amplitude})", lw=2)
    ax.plot(x, norm_pdf, label="normalised (peak=1)", lw=2)

    ax.set_title(f"mu={mu}, gamma={gamma}")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("value")
plt.tight_layout()
plt.savefig("lorentzians.png", dpi=150)
plt.show()