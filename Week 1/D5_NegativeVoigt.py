import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from pymultifit.fitters.backend import BaseFitter
from pymultifit.fitters.utilities_f import sanity_check

np.random.seed(42)
x = np.linspace(-10, 10, 2000)


def raw_voigt(x, amplitude, x0, sigma, gamma):

    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = np.real(wofz(z))
    peak = np.real(wofz(1j * gamma / (sigma * np.sqrt(2))))
    return amplitude * profile / peak


class VoigtFitter(BaseFitter):


    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 4

    def fit_boundaries(self):
        lb = (0, -np.inf, 0, 0)
        ub = (np.inf, np.inf, np.inf, np.inf)
        return lb, ub

    @staticmethod
    def fitter(x, params):
        return raw_voigt(x, *params)


class UnboundedAmpVoigtFitter(VoigtFitter):


    def fit_boundaries(self):
        lb = (-np.inf, -np.inf, 0, 0)
        ub = (np.inf, np.inf, np.inf, np.inf)
        return lb, ub



y_peak = raw_voigt(x, amplitude=3.0, x0=-3.0, sigma=0.8, gamma=0.5)
y_dip = raw_voigt(x, amplitude=-2.5, x0=4.0, sigma=0.9, gamma=0.7)
noise = np.random.normal(0, 0.1, size=x.shape)
y = y_peak + y_dip + noise

normal_fitter = VoigtFitter(x, y)
normal_fitter.fit(p0=[(3.0, -3.0, 0.8, 0.5), (2.5, 4.0, 0.9, 0.7)])

unbounded_fitter = UnboundedAmpVoigtFitter(x, y)
unbounded_fitter.fit(p0=[(3.0, -3.0, 0.8, 0.5), (-2.5, 4.0, 0.9, 0.7)])

print("Normal fitter params (amplitude >= 0):\n", normal_fitter.get_model_parameters())
print("Unbounded-amplitude fitter params:\n", unbounded_fitter.get_model_parameters())


fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(x, y, s=4, color="gray", alpha=0.4, label="Data (peak + dip + noise)")
ax.plot(x, normal_fitter.get_fitted_curve(), color="crimson", lw=2, label="Normal fitter (amplitude >= 0)")
ax.plot(x, unbounded_fitter.get_fitted_curve(), color="royalblue", lw=2, label="Unbounded-amplitude fitter")
ax.axhline(0, color="black", lw=0.5)
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Normal vs. unbounded-amplitude VoigtFitter")
plt.show()