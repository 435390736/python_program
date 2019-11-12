import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from time import time

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris["DESCR"])  # 数据集描述信息
# print(iris["feature_names"])  # 特征名

X = iris["data"][:, 3:]  # 取花瓣宽度这个维度
# print(X)

# print(iris["target"])  # 预测的分类号
y = iris["target"]

log_reg = LogisticRegression(multi_class="ovr", solver="sag")  # 创建逻辑回归模型
log_reg.fit(X, y)  # 训练模型

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # 测试集,(0,3)平均分为1000个点
# print(X_new)

y_proba = log_reg.predict_proba(X_new)  # 测试集的概率
y_hat = log_reg.predict(X_new)  # 测试集的分类
print(y_proba)
print(y_hat)
print("w1", log_reg.coef_)
print("w0", log_reg.intercept_)

plt.plot(X_new, y_proba[:, 2], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 1], "r-", label="Iris-Versicolour")
plt.plot(X_new, y_proba[:, 0], "b--", label="Iris-Setosa")
plt.show()
