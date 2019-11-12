"""
1.初始化theta
2.求梯度gradients（求导）
3.调整theta ： theta = theta-grad * learning_rate
  learning_rate为学习率，学习率为正(超参数:如何使模型求解的更快更好，太小会使迭代次数多费时，太大会不准确)
4.当梯度近似为0时停止迭代（阈值，threshold,也为超参数）
"""
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
# print(X_b)

learning_rate = 0.1  # 学习率
n_literations = 10000  # 总迭代次数
m = 100  # 一百个样本

# 1.初始化
theta = np.random.randn(2, 1)  # 两行一列的矩阵,[w0, w1]
count = 0

# 4.直接设置超参数迭代次数(若设置阈值，当迭代次数很大时时间会很长),迭代次数到了就认为收敛了
for iteration in range(n_literations):
    count += 1
    # 2.求梯度gradients(此处是向量不是单一的值)
    # 乘1/m是为了求损失函数的均值(随着数据量的增加而更准确)
    gradients = 1/m * X_b.T.dot(X_b.dot(theta)-y)
    # 3.调整theta
    theta = theta - learning_rate * gradients

print(count)
print(theta)
