import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv("./insurance.csv")
# print(type(data))
# print(data.head())
# print(data.tail())
# print(data.describe())

# data_count = data["age"].value_counts()
# print(data_count)
# data_count[:10].plot(kind="bar")
# plt.show()

for i in range(len(data)):
    data.loc[i, "sex"] = 1 if data.loc[i, "sex"] == "male" else 2
    data.loc[i, "smoker"] = 1 if data.loc[i, "sex"] == "male" else 2
    if data.loc[i, "region"] == "southwest":data.loc[i, "region"] = 1
    elif data.loc[i, "region"] == "northwest":data.loc[i, "region"] = 2
    elif data.loc[i, "region"] == "southeast":data.loc[i, "region"] = 3
    else:data.loc[i, "region"] =4

plt.figure(figsize=(15, 8), dpi=80)
plt.plot(data["age"],data["charges"], "b.", alpha=1)


d = {1: "g1", 3: "r+", 5: "y2"}
for i in d:
    poly_features = PolynomialFeatures(degree=i, include_bias=False)
    X_poly = poly_features.fit_transform(data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']])

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, data["charges"])
    y_preidict = lin_reg.predict(X_poly)

    plt.plot(X_poly[:, 0], y_preidict, d[i], alpha=0.8)
plt.show()
