import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 方法一
ridge_reg = Ridge(alpha=1, solver="auto")  # alpha是惩罚项中的alpha,slover是自动选择
ridge_reg.fit(X, y)
"""
由于在新版的sklearn中，所有的数据都应该是二维矩阵，哪怕它只是单独一行或一列
（比如仅仅只用了一个样本数据），所以需要使用.reshape(1,-1)进行转换
"""
print(ridge_reg.predict(np.array(1).reshape(1, -1)))  # 预测值
print(ridge_reg.intercept_)  # 截距
print(ridge_reg.coef_)  # 参数

# 方法二
# penalty是惩罚函数,可选l1或l2(默认为l2),max_iter是最大迭代次数(可以不填默认1000)
sgd_reg = SGDRegressor(penalty="l2", max_iter=10000)
sgd_reg.fit(X, y.ravel())  # ravel函数把y由列向量变为行向量

print(sgd_reg.predict(np.array(1).reshape(1, -1)))  # 预测值
print(sgd_reg.intercept_)  # 截距
print(sgd_reg.coef_)  # 参数
