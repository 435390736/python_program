from sklearn import datasets
from sklearn.linear_model import LogisticRegression


iris = datasets.load_iris()
X = iris["data"]  # 所有行所有列,4个维度
y = iris["target"]

log_reg = LogisticRegression(multi_class="ovr", solver="sag", max_iter=10000)  # 创建逻辑回归模型
log_reg.fit(X, y)  # 训练模型

print("w1", log_reg.coef_)
print("w0", log_reg.intercept_)
