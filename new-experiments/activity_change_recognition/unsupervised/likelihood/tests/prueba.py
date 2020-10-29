import sys
import numpy as np
from scipy.stats import multivariate_normal
sys.path.append('../densratio')
from core import densratio

np.set_printoptions(threshold=sys.maxsize)
np. set_printoptions(suppress=True)

np.random.seed(1)
x = multivariate_normal.rvs(size=35, mean=[1, 1], cov=[[1. / 8, 0], [0, 1. / 8]])
y = multivariate_normal.rvs(size=35, mean=[1, 1], cov=[[1. / 2, 0], [0, 1. / 2]])

print(x)
print(y)

alpha = 0.01
densratio_obj = densratio(x, y, alpha=alpha, sigma_range=[0.1, 0.3, 0.5, 0.7, 1], lambda_range=[0.01, 0.02, 0.03, 0.04, 0.05])
print(densratio_obj)