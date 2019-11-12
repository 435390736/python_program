import numpy as np
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

lin_reg = LinearRegression()  # 创建模型对象
lin_reg.fit(X, y)  # 训练(求解模型),fit用的是解析解的方式求解
print(lin_reg.intercept_, lin_reg.coef_)  # intercept_为截距,coef_为参数

X_new = np.array([[0], [2]])
print(lin_reg.predict(X_new))
