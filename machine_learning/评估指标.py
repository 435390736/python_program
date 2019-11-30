from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier  # SGD算法分类
from sklearn.model_selection import StratifiedKFold  # k折交叉验证
from sklearn.base import clone
from sklearn.model_selection import cross_val_score  # 交叉验证及分数
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict  # 交叉验证及预测类别
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.metrics import precision_score  # 精确率
from sklearn.metrics import recall_score  # 召回率
from sklearn.metrics import f1_score  # f1-score
from sklearn.metrics import precision_recall_curve  # 精确率及召回率
from sklearn.metrics import roc_curve  # roc曲线
from sklearn.metrics import roc_auc_score  # auc面积
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
import warnings# 忽略警告
warnings.filterwarnings("ignore")

mnist = fetch_mldata("MNIST original", data_home="test_data_home")
# print(mnist)

X, y = mnist["data"], mnist["target"]
print(X.shape, y.shape)

some_digit = X[36000]  # 取数据集中第36000张图片
# print(some_digit)
some_digit_image = some_digit.reshape(28, 28)
# print(some_digit_image)

# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)  # 生成60000个随机索引,打乱数据集的顺序
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]  # 一次性取值

# 做二分类,把是5的和不是5的分开
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# print(y_test_5)

# 逻辑回归分类器
sgd_clf = SGDClassifier(loss="log", random_state=42, max_iter=10000, tol=1e-4)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))  # 预测第36000张图片的分类结果

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


# 方法一实现3折交叉验证
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):  # 切分数据得到索引
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]  # 训练集
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]  # 验证集
    y_test_folds = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    print(y_pred)
    n_correct = sum(y_pred == y_test_folds)
    print(n_correct / len(y_pred))

方法二实现3折交叉验证(不能选择超参数)
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="precision"))
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="recall"))

# 自定义一个假的分类器
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X_train), 1), dtype=bool)

# 方法三实现3折交叉验证
never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# 方法四实现3折交叉验证(混淆矩阵)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(confusion_matrix((y_train_5, y_train_pred)))
# 理想情况完美的混淆矩阵的结果
y_train_perfect_prediction = y_train_5
print(confusion_matrix(y_train_5, y_train_perfect_prediction))


print(precision_score(y_train_5, y_train_pred))  # 精确率
print(recall_score(y_train_5, y_train_pred))  # 准确率
print(sum(y_train_pred))
print(f1_score(y_train_5, y_train_pred))  # f1_score


sgd_clf.fit(X_train, y_train_5)
y_scores = sgd_clf.decision_function([some_digit])  # decision_function:求模型的决策边界
print(y_scores)  # 越小越接近决策边界,越大越好(分类越有信心)


threshold = 0  # 阈值,相当于调节z(0~1)
y_some_digit_pred = (y_scores > threshold)  # 大于阈值为1,小于阈值为0
print(y_some_digit_pred)

threshold = 200000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
print(y_scores)
# precision_recall曲线
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
print(precisions, recalls, thresholds)


def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "r--", label="recall")
    plt.xlabel("threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
plt.show()


y_train_pred_90 = (y_scores > 70000)
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))


fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


# 绘制roc曲线
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


plot_roc_curve(fpr, tpr)
plt.show()
# 打印roc曲线的面积
print(roc_auc_score(y_train_5, y_scores))


# 逻辑回归与随机森林的对比
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, "b-", label="SGD")
plt.plot(fpr_forest, tpr_forest, label="Random Forest")
plt.legend(loc="upper right")
plt.show()
print(roc_auc_score(y_train_5, y_scores_forest))
