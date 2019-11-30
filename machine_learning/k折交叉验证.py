import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from time import time

iris = datasets.load_iris()
X = iris["data"][:, 3:]
y = iris["target"]


# 打印k折交叉验证后不同超参数对应的结果
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:0.3f}(std: {1:.3f})".format(
                results["mean_test_score"][candidate],
                results["std_test_score"][candidate]))
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


start_time = time()
param_grid = {"tol": [1e-4, 1e-3, 1e-2],
              "C": [0.4, 0.6, 0.8]}
log_reg = LogisticRegression(multi_class="ovr", solver="sag")
grid_search = GridSearchCV(log_reg, param_grid=param_grid, cv=3)  # 3折
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d canditate parameter settings"
      % (time()-start_time, len(grid_search.cv_results_["params"])))
report(grid_search.cv_results_)

X_new = np.linspace(0, 4, 1000).reshape(-1, 1)
y_proba = grid_search.predict_proba(X_new)
y_hat = grid_search.predict(X_new)
print("y_proba\n", y_proba)
print("y_hat\n", y_hat)

plt.plot(X_new, y_proba[:, 2], "g-", label="Virginica")
plt.plot(X_new, y_proba[:, 1], "r-", label="Versicolour")
plt.plot(X_new, y_proba[:, 0], "b--", label="Setosa")
plt.show()
