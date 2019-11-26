import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier  # 决策树类别
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor   #决策树回归
from sklearn.model_selection import train_test_split  # 切分训练集和测试集的比例
from sklearn.metrics import accuracy_score  # 评估准确率
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = load_iris()
data = pd.DataFrame(iris.data)
data.columns = iris.feature_names
data["Species"] = load_iris().target
# print(data)
# print(data.describe())

x = data.iloc[:, :2]  # 花萼的长度和宽度
y = data.iloc[:, -1]  # 花的种类
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=45)
# 75%的数据为训练集,25%为测试集,随机种子为42(写死的话每次切割的数据集会一样)

tree_clf = DecisionTreeClassifier(max_depth=2, criterion="entropy")  # 创建决策树模型,max_depth表示树的层次最大为2(避免过拟合),criterion为分割标准
tree_clf.fit(x_train, y_train)
y_test_hat = tree_clf.predict(x_test)
print("acc score:", accuracy_score(y_test, y_test_hat))  # 准确度

print(tree_clf.predict_proba(np.array([5, 1.5]).reshape(1, -1)))  # 预测概率
print(tree_clf.predict(np.array([5, 1.5]).reshape(1, -1)))  # 预测类别号

depth = np.arange(1, 15)
err_list = []
for d in depth:
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=d)
    clf.fit(x_train, y_train)
    y_test_hat = clf.predict(x_test)
    result = (y_test_hat == y_test)
    if d == 1:
        print(result)   # 求准确率,与前面acc一样
    err = 1 - np.mean(result)  # 错误率
    print(100 * err)
    err_list.append(err)
    print(d, "错误率:%.2f%%" % (100 * err))

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体为中文黑体
plt.figure(facecolor="w")  # 图的底色为白色
plt.plot(depth, err_list, "ro-", lw=2)
plt.xlabel("决策树深度", fontsize=15)
plt.ylabel("错误率", fontsize=15)
plt.title("决策树深度和过拟合", fontsize=15)
plt.grid(True)
plt.show()