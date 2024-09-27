import numpy as np
import matplotlib.pyplot as plt 

from nngp import NNGP

L = 20
n_training = 11
n_test = 30
sigma_w_2 = 2
sigma_b_2 = 1
sigma_eps = 0.2

def signal(x):
    return 5*np.cos(0.5*x)
    # t1 = 20
    # t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))
    # t3 = np.e
    # t4 = - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
    # return t1 + t2 + t3 + t4

training_data = np.linspace(-10, 10, n_training).reshape((n_training, 1))
training_output = signal(training_data)
training_targets = training_output + sigma_eps * np.random.randn(n_training, 1)

test_data = np.linspace(-10, 10, n_test).reshape(n_test, 1)

regression = NNGP(
    training_data,
    training_targets,
    test_data, L,
    sigma_b_2=sigma_b_2,
    sigma_w_2=sigma_w_2,
    sigma_eps_2=sigma_eps**2)

regression.train()

predictions, covariance = regression.predict()
variances = np.diag(covariance)

plt.scatter(training_data, training_targets, label='train')
plt.plot(test_data, signal(test_data), label='signal', ls='--', color='black', alpha=.7, lw=1)
# predictions = predictions.flatten()
# test_data = test_data.flatten()
# plt.errorbar(test_data, predictions, yerr=variances, color='red', label='prediction')
plt.legend()
plt.savefig('1D_regression')
