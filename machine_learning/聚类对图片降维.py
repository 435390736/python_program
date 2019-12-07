from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings# 忽略警告
warnings.filterwarnings("ignore")


# restore_image函数用于恢复图像,cluster是聚类预测的结果,cb是聚类的中心点,shape原始图像的形状
def restore_image(cb, cluster, shape):
    row, col, dumpy = shape
    image = np.empty((row, col, 3))
    index = 0  # 聚类类别,共256个类别
    for r in range(row):  # r,c分别是图像的行和列坐标点
        for c in range(col):
            image[r, c] = cb[cluster[index]]
            index += 1
    return image


def show_scatter(a):
    N = 10
    print("原始数据:\n", a)
    density, edges = np.histogramdd(a, bins=[N, N, N], range=[(0, 1), (0, 1), (0, 1)])
    density /= density.max()
    x = y = z = np.arange(N)
    d = np.meshgrid(x, y, z)

    # 统计像素点的像素值
    fig = plt.figure(1, facecolor="w")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(d[1], d[0], d[2], c="r", s=100*density, marker="o", depthshade=True)
    ax.set_xlabel(u"红色分量")
    ax.set_ylabel(u"绿色分量")
    ax.set_zlabel(u"蓝色分量")
    plt.title(u"图像颜色三维频数分布", fontsize=20)

    plt.figure(2, facecolor="W")
    den = density[density > 0]
    den = np.sort(den)[::-1]
    t = np.arange(len(den))
    plt.plot(t, den, "r-", t, den, "go", lw=2)
    plt.title(u"图像颜色频数分布", fontsize=18)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    matplotlib.rcParams["font.sans-serif"] = [u'SimHei']
    matplotlib.rcParams["axes.unicode_minus"] = False

    num_vq = 25  # 25个颜色类别,即最后降到25个维度
    im = Image.open("kaka.jpg")  # 读入图片
    image = np.array(im).astype(np.float) / 255  # 转为numpy数组
    image = image[:, :, :3]  # 选择所有行所有列和rgb(透明度alpha不保存)
    image_v = image.reshape((-1, 3))  # 从三维转为二维数据
    model = KMeans(num_vq)  # 没有开始聚类,num_vq是类别总数
    # show_scatter(image_v)  # 统计图像像素点数据

    N = image_v.shape[0]  # 图像像素总数,把第一列取出来统计有多少条样本
    print(N)
    print(image_v.shape)
    # 从原始样本中随机选择足够多的样本(比如1000个),计算聚类中心
    idx = np.random.randint(0, N, size=1000)
    image_sample = image_v[idx]
    model.fit(image_sample)  # 从1000个样本中选取最重要的25个样本作为中心点,按像素颜色聚类
    c = model.predict(image_v)  # 预测聚类结果
    print("聚类结果:\n", c)
    print("聚类中心:\n", model.cluster_centers_)

    plt.figure(figsize=(15, 8), facecolor="w")
    plt.subplot(121)
    plt.axis("off")
    plt.title(u"原始图片", fontsize=18)
    plt.imshow(image)

    plt.subplot(122)
    # c是聚类预测的结果,model.cluster_centers_是聚类的中心点,image.shape原始图像的形状
    vq_image = restore_image(model.cluster_centers_, c, image.shape)
    plt.axis("off")
    plt.title(u"矢量量化后图片:%d色" % num_vq, fontsize=20)
    plt.imshow(vq_image)

    plt.tight_layout(1.2)
    plt.show()
