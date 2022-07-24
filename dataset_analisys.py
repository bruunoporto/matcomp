import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dataset():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    print(df)
    X = df.iloc[0:100, [0, 2]].values
    # plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # plt.xlabel('petal length')
    # plt.ylabel('sepal length')
    # plt.legend(loc='upper left')
    # plt.show()
    x = []
    for el in X:
        l = [float(el[0]), float(el[1])]
        x.append(l)
    y_t = []
    for el in y:
        y_t.append(float(el))
    return x, y_t