import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

lasso_reg = Lasso(alpha=0.15)
lasso_reg.fit(X, y)
print(lasso_reg.predict(np.array(1.5).reshape(1, -1)))
print(lasso_reg.coef_)
print(lasso_reg.intercept_)

sgd_reg = SGDRegressor(penalty="l1", max_iter=10000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict(np.array(1.5).reshape(1, -1)))
print(sgd_reg.coef_)
print(sgd_reg.intercept_)
