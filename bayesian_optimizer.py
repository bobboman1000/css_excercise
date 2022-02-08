import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import sklearn.gaussian_process as gp


def bayesian_optimization(X, y, n_iterations, known_param_values, known_result_values, acquisition_function, length_scale=1, random_state=None):
    """
    Implementierung von Bayesian optimization.

    Fehlerbehaftet aufgrund von Numerischen Fehlern.

    :param X:
    :param y:
    :param n_iterations:
    :param known_param_values:
    :param known_result_values:
    :param acquisition_function:
    :param random_state:
    :return: Darstellung des Gauß-Prozesses
    """

    #  Initiate GP with some kerne, RBF as most common
    rbf_kernel = gp.kernels.RBF(length_scale=length_scale)
    gaussian_process = gp.GaussianProcessRegressor(kernel=rbf_kernel, random_state=random_state, alpha=1e-10, n_restarts_optimizer=3)

    x_pts, y_pts = np.asarray(known_param_values), np.asarray(known_result_values)
    x_pts: np.ndarray
    y_pts: np.ndarray

    for n in range(n_iterations):
        x_pts, y_pts = x_pts.reshape(-1, 1), y_pts.reshape(-1, 1)

        # Fit GP
        gaussian_process.fit(X=x_pts, y=y_pts)

        # Sample function
        x_lin = np.linspace(0.001, 10, 200).reshape(-1, 1)
        y_pred, y_pred_std = gaussian_process.predict(x_lin, return_std=True)
        y_pred = y_pred.flatten()

        # Compute argmax in acquisition function
        best_observed = np.max(y_pts)
        next_sample_idx = np.argmax(acquisition_function(best_observed, y_pred, y_pred_std))
        next_sample = x_lin[next_sample_idx][0]

        # Evaluate sampled param
        e = SVC(C=next_sample)

        augmented_model = e.fit(X, y)
        augmented_model_score = cross_val_score(augmented_model, X, y, cv=10).mean()

        x_pts = np.append(x_pts, next_sample)
        y_pts = np.append(y_pts, augmented_model_score)

        fig = plt.figure(figsize=(50,5))
        ax = fig.add_subplot(111)
        ax.plot(x_lin, y_pred, linewidth=5)
        ax.scatter(x_pts[:-1], y_pts[:-1], s=200)
        ax.scatter(x_pts[-1], y_pts[-1], s=200, color="red")
        ax.fill_between(x_lin.flatten(), y_pred-y_pred_std, y_pred+y_pred_std, alpha=0.2)
        plt.show()


def positive_probability_of_improvement(best_observed, y, y_std):
    poi = lambda mean_a, mean_b, sigma_b: np.log(1 - norm.cdf(mean_a, loc=mean_b, scale=sigma_b))
    return [poi(best_observed, y_i, std_y_i) for y_i, std_y_i in zip(y, y_std)]