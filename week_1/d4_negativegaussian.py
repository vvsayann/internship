import numpy as np
import matplotlib.pyplot as plt
from pymultifit.fitters.backend.baseFitter import BaseFitter

np.random.seed(42)
x = np.linspace(-10, 10, 2000)


def raw_lorentzian(x, amplitude, mu, gamma):
    return amplitude * (gamma ** 2) / ((x - mu) ** 2 + gamma ** 2)


y_peak = raw_lorentzian(x, amplitude=3.0, mu=-3.0, gamma=1.0)
y_dip = raw_lorentzian(x, amplitude=-2.5, mu=4.0, gamma=1.2)
noise = np.random.normal(0, 0.15, size=x.shape)
y = y_peak + y_dip + noise


class LorentzianFitter(BaseFitter):

    def __init__(self, x_values, y_values, max_iterations=1000):
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    def fit_boundaries(self):
        lb = (0, -np.inf, 0)
        ub = (np.inf, np.inf, np.inf)
        return lb, ub

    @staticmethod
    def fitter(x, params):
        amplitude, mu, gamma = params
        return amplitude * (gamma ** 2) / ((x - mu) ** 2 + gamma ** 2)


class LorentzianFitterNegativeAmplitude(LorentzianFitter):
    def fit_boundaries(self):
        lb = (-np.inf, -np.inf, 0)
        ub = (0, np.inf, np.inf)
        return lb, ub


normal_fitter = LorentzianFitter(x, y)
normal_fitter.fit(p0=[(3.0, -3.0, 1.0), (2.5, 4.0, 1.2)])

neg_fitter = LorentzianFitterNegativeAmplitude(x, y)
neg_fitter.fit(p0=[(3.0, -3.0, 1.0), (-2.5, 4.0, 1.2)])

print("Normal fitter params:\n", normal_fitter.get_model_parameters())
print("Negative-capable fitter params:\n", neg_fitter.get_model_parameters())


fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(x, y, s=4, color="gray", alpha=0.4, label="Data (peak + dip + noise)")
ax.plot(x, normal_fitter.get_fitted_curve(), color="crimson", lw=2, label="Normal fitter (amplitude >= 0)")
ax.plot(x, neg_fitter.get_fitted_curve(), color="royalblue", lw=2, label="Negative-capable fitter (unbounded amplitude)")
ax.axhline(0, color="black", lw=0.5)
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Normal vs. negative-amplitude-capable LorentzianFitter")
plt.tight_layout()
plt.show()