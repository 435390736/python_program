import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans  # 导入KMeans
from sklearn.cluster import MiniBatchKMeans  # 只选择一部分求均值,速度快

# 用于画图
def expend(a, b):
    d = (b - a) * 0.1
    return a-d, b+d


if __name__ == "__main__":
    N = 400  # 创建400个样本
    centers = 4  # 4个类别
    # 创建聚类模拟数据(方差相同)
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    # 方差cluster_std越大数据点越分散,越小数据越密集
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, random_state=2, cluster_std=(1, 2.5, 0.5, 2))
    # 从每个类别中选取样本的数量不同时分类的情况
    data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)

    cls = MiniBatchKMeans(n_clusters=4, init="k-means++")
    y_hat = cls.fit_predict(data)
    y2_hat = cls.fit_predict(data2)
    y3_hat = cls.fit_predict(data3)

    m = np.array(((1, 1), (1, 3)))
    data_r = data.dot(m)  # 点成矩阵m,得到拉伸旋转后的数据矩阵
    y_r_hat = cls.fit_predict(data_r)

    matplotlib.rcParams["font.sans-serif"] = [u'SimHei']
    matplotlib.rcParams["axes.unicode_minus"] = False
    cm = matplotlib.colors.ListedColormap(list("rgbm"))  # 四个类别数据点的颜色

    plt.figure(figsize=(9, 10), facecolor="w")
    plt.subplot(421)
    plt.title(u"原始数据")
    # c代表类别,cmap代表不同类别点的颜色
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors="none")
    # 获取x1,x2最大,最小数值
    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expend(x1_min, x1_max)
    x2_min, x2_max = expend(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(422)
    plt.title(u"KMeans++聚类")
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors="none")
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(423)
    plt.title(u"旋转后的数据")
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors="none")
    x1_min, x2_min = np.min(data_r, axis=0)
    x1_max, x2_max = np.max(data_r, axis=0)
    x1_min, x1_max = expend(x1_min, x1_max)
    x2_min, x2_max = expend(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(424)
    plt.title(u"旋转后KMeans++聚类")
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors="none")
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(425)
    plt.title(u"方差不相等数据")
    plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors="none")
    x1_min, x2_min = np.min(data2, axis=0)
    x1_max, x2_max = np.max(data2, axis=0)
    x1_min, x1_max = expend(x1_min, x1_max)
    x2_min, x2_max = expend(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(426)
    plt.title(u"方差不相等KMeans++聚类")
    plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors="none")
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(427)
    plt.title(u"数量不相等数据")
    plt.scatter(data3[:, 0], data3[:, 1], c=y3, s=30, cmap=cm, edgecolors="none")
    x1_min, x2_min = np.min(data3, axis=0)
    x1_max, x2_max = np.max(data3, axis=0)
    x1_min, x1_max = expend(x1_min, x1_max)
    x2_min, x2_max = expend(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(428)
    plt.title(u"数量不相等KMeans++聚类")
    plt.scatter(data3[:, 0], data3[:, 1], c=y3, s=30, cmap=cm, edgecolors="none")
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.tight_layout(2, rect=(0, 0, 1, 0.97))
    plt.suptitle(u"数据分布对KMeans聚类的影响", fontsize=18)
    plt.show()
