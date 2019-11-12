import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

elastic_net = ElasticNet(alpha=0.0001, l1_ratio=0.15)
elastic_net.fit(X, y)
print(elastic_net.predict(np.array(1.5).reshape(1, -1)))
print(elastic_net.coef_)
print(elastic_net.intercept_)

sgd_reg = SGDRegressor(penalty="elasticnet", max_iter=10000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict(np.array(1.5).reshape(1, -1)))
print(sgd_reg.coef_)
print(sgd_reg.intercept_)
