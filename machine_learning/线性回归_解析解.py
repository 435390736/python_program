import numpy as np
import matplotlib.pyplot as plt

# 这里相当于使随机X维度X1,rand是随机均匀分布
X = 2 * np.random.rand(100, 1)  # 一百行一列,[0,2)
# 人为的设置真实的Y一列,np.random.randn(100,1)是设置误差,randn是标准正态分布
y = 4 + 3 * X + np.random.randn(100, 1)  # 一百行一列
# 整合X0和X1,np.c_是(combine)连接两个向量
X_b = np.c_[np.ones((100, 1)), X]  # 两个维度(X0=1.X1=X)
# print(X_b)

# 常规等式求解theta,linalg是线性代数模块,inv是求逆，dot是点积(点乘)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

# 创建测试集里面的X1
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)

# 绘图
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
