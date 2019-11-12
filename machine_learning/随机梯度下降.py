import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
# print(X_b)

n_epochs = 500
t0, t1 = 5, 50  # 超参数

m = 100  # 100行数据


# 让学习率随着迭代次数增加而变小
def learning_schedule(t):
    return t0 / (t + t1)


# 1.随机初始化theta
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):  # 学习的轮次(迭代次数iterations)
    for i in range(m):  # 从100行数据中随机抽取一条
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        learning_rate = learning_schedule(epoch*m + i)
        theta = theta - learning_rate * gradients

print(theta)
