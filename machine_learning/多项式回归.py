import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3  # 范围(-3, 3)
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, "b.")

d = {1: "g-", 2: "r+", 5: "y1"}
for i in d:
    # degree是几阶变化,include_bias表示是否添加W0
    poly_features = PolynomialFeatures(degree=i, include_bias=False)
    X_poly = poly_features.fit_transform(X)  # 对原来数据集进行转换
    # print(X[0])
    # print(X_poly[0])
    # print(X_poly[:, 0])

    # LinearRegression()中有默认参数fit_intercept=True会自动添加上截距
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    print(lin_reg.coef_, lin_reg.intercept_)
    y_predict = lin_reg.predict(X_poly)
    plt.plot(X_poly[:, 0], y_predict, d[i])

plt.show()
