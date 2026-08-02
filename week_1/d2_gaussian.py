import numpy as np
import matplotlib.pyplot as plt

# loc scale family for gaussian distribution
def gaussian_pdf(x, loc, scale):
    """
    x : array_like
        Points at which we will calculate.
    loc : float
        Location parameter (mean, mu).
    scale : float
        Defines the parameter for the scale (sigma).

    Returns array_like
        Gaussian distribution evaluated at x (area = 1).
    """
    z = (x - loc) / scale
    return np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * scale)

# gaussian function
def gaussian(x, amplitude, mu, sigma):
    """
    x : array_like
        Input wavelength (or frequency) values.
    amplitude : float
        Peak height of the distribution.
    mu : float
        Location of the peak.
    sigma : float
        Determines how spread out the distribution is.

    Returns array_like
        Gaussian profile evaluated at x (peak = amplitude).
    """
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


# parameters for three distribtuions ( mu and sigma values)
params = [
    (0, 1),
    (-3, 3),
    (0.5, 2),
]
amplitude = 2
x = np.linspace(-20, 20, 2000)

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)

# loop for the plot to run on all three
for (mu, sigma), ax in zip(params, axes):
    pdf = gaussian_pdf(x, mu, sigma)
    norm_pdf = amplitude * (pdf / np.max(pdf))
    main = gaussian(x, amplitude, mu, sigma)

    ax.plot(x, pdf, label="gaussian_pdf (area=1)", lw=2)
    ax.plot(x, main, label=f"gaussian (amp={amplitude})", lw=2)
    ax.plot(x, norm_pdf, label="normalised (peak=amp)", lw=2)

    ax.set_title(f"amp={amplitude}, mu={mu}, sigma={sigma}")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("value")
plt.tight_layout()
plt.savefig("gaussians.png", dpi=150)
plt.show()