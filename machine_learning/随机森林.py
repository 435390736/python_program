from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier  # bagging:并行学习
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]  # 花萼长度和宽度
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rnd_clf = RandomForestClassifier(n_estimators=50, max_leaf_nodes=16, n_jobs=1)
# n_estimators表示决策树数量(越多越准,太多会过拟合), max_leaf_nodes控制树的复杂程度,n_jobs表示多少个线程(-1表示有多少线程就用多少线程)
rnd_clf.fit(X_train, y_train)

# 与上面直接创建RandomForestClassifier是等价的,max_samples是训练时是否使用全部样本
# splitter表示每棵小树使用全部样本,但只选取一部分维度(重点在于随机维度而不是随机样本)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
    n_estimators=50, max_samples=1.0, bootstrap=True, n_jobs=1
)
bag_clf.fit(X_train, y_train)


y_pred_rf = rnd_clf.predict(X_test)
y_pred_bag = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))
print(accuracy_score(y_test, y_pred_bag))

# 用随机森林选取表较重要的特征(相关系数越大(越相关),该特征越重要)
# 可用皮尔逊相关系数,L1L2正则,树寻找
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
