from sklearn.neural_network import MLPClassifier  # 多层感知机

X = [[0., 0.], [1., 1.]]  # 两条数据，每条数据两个维度（两个输入神经元）
y = [0, 1]  # 一个输出神经元（输出只能是0或1）

clf = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=[5, 2], max_iter=2000,tol=1e-4)
# 两个隐藏层，分别有5个，2个神经元
clf.fit(X, y)

predicted_value = clf.predict([[2., 2.], [-1., -2.]])
print("预测值", predicted_value)
predicted_proba = clf.predict_proba([[2., 2.], [-1., -2.]])
print("预测概率", predicted_proba)
print("参数的形状", [coef.shape for coef in clf.coefs_])
print("参数", [coef for coef in clf.coefs_])