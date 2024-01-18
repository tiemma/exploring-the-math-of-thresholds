import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


# How well does our fit work with the function
def goodness_of_fit(observed, expected):
    chisq = np.sum(((observed - expected) / np.std(observed)) ** 2)

    # Present the chisq value percentage relative to the sample length
    n = len(observed)
    return ((n - chisq) / n) * 100


# This is the function we are trying to fit to the data.
def func(x, a, b, c):
    result = a * np.sin(2 * b * x) + c
    # Make the data entirely positive
    result[result < 0] = 0
    return result


fig, ax = plt.subplots()

# Generate some random data
xdata = np.linspace(0, 10, 50)
y = func(xdata, 2.5, 1.3, 0.5)
y_noise = np.abs(np.random.normal(size=xdata.size))
ydata = y + y_noise

# Matplot lib settings
plt.grid(True)
plt.autoscale()
plt.tight_layout()

# Plot the actual data
ax.plot(xdata, ydata, label="Normal");

# Graph plot without the fit
ax.set_ylabel("User Sales (in millions)")
ax.set_xlabel("Hour")
ax.set_title('Curve Fitting')
fig.savefig(f"./curve_fitting_without_fit.png", bbox_inches="tight")

# The actual curve fitting happens here
optimizedParameters, pcov = opt.curve_fit(func, xdata, ydata)
best_fit_data = func(xdata, *optimizedParameters)

# Use the optimized parameters  to plot the best fit
print(f"Fit, std_dev: {np.std(ydata - best_fit_data)}, fit_percent: {goodness_of_fit(ydata, best_fit_data)}")
ax.plot(xdata, best_fit_data,
        label=f"Fit, std_dev: {np.std(ydata - best_fit_data)}, fit_percent: {goodness_of_fit(ydata, best_fit_data)}")
ax.legend(loc='lower left', prop={'size': 7}, borderpad=0.5, labelspacing=0)
if np.std(ydata - best_fit_data) > 1:
    fig.savefig(f"./curve_fitting_with_unfit.png", bbox_inches="tight")
else:
    fig.savefig(f"./curve_fitting_with_fit.png", bbox_inches="tight")

# Graph plot with the difference in the values
ax2 = ax.twinx()
ax2.plot(xdata, np.abs(ydata - best_fit_data), label="Difference", fillstyle="left", color="black")
ax2.legend(loc='lower right', prop={'size': 7}, borderpad=0.5, labelspacing=0)
ax2.set_ylabel("Normal - Fit")
ax2.axhline(y=2.0, label=f"Difference threshold", linestyle="--")
fig.savefig(f"./curve_fitting_with_fit_difference.png", bbox_inches="tight")
