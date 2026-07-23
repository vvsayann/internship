import numpy as np
import matplotlib.pyplot as plt
from pymultifit.fitters.backend import BaseFitter
from pymultifit.fitters.utilities_f import sanity_check

np.random.seed(42)
x = np.linspace(-10, 10, 2000)


def raw_lorentzian(x, amplitude, x0, gamma):

    return amplitude * (gamma ** 2) / ((x - x0) ** 2 + gamma ** 2)


class LorentzianFitter(BaseFitter):


    def __init__(self, x_values, y_values, max_iterations: int = 1000):
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    def fit_boundaries(self):
            lb = (0, -np.inf, 0)
            ub = (np.inf, np.inf, np.inf)
            return lb, ub

    @staticmethod
    def fitter(x, params):
        return raw_lorentzian(x, *params)


class UnboundedAmpLorentzianFitter(LorentzianFitter):

    def fit_boundaries(self):
        lb = (-np.inf, -np.inf, 0)
        ub = (np.inf, np.inf, np.inf)
        return lb, ub



y_peak = raw_lorentzian(x, amplitude=3.0, x0=-3.0, gamma=1.0)
y_dip = raw_lorentzian(x, amplitude=-2.5, x0=4.0, gamma=1.2)
noise = np.random.normal(0, 0.1, size=x.shape)
y = y_peak + y_dip + noise

normal_fitter = LorentzianFitter(x, y)
normal_fitter.fit(p0=[(3.0, -3.0, 1.0), (2.5, 4.0, 1.2)])

unbounded_fitter = UnboundedAmpLorentzianFitter(x, y)
unbounded_fitter.fit(p0=[(3.0, -3.0, 1.0), (-2.5, 4.0, 1.2)])

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
ax.set_title("Normal vs. unbounded-amplitude LorentzianFitter")
plt.show()