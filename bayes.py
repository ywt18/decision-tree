import random
import numpy as np
from math import e
from math import pow
import math
from sklearn.datasets import load_iris

# 获得数据及初始化参数
iris = load_iris()
n_total, n_feature = iris.data.shape
n_train = 120;
n_test = 30;
n_target = 3;

book = np.zeros(150, dtype=int)

# 每五个样本中，选一个作为测试样本
for i in range(n_test):
    val = 5 * i + random.randint(0, 4)
    book[val] = 1

# 初始化训练样本和测试样本
data_train = np.zeros((n_total, 5))
data_test = np.zeros((n_total, 5))
cnt1 = 0;
cnt2 = 0

# 生成训练样本和测试样本
for i in range(n_total):
    if book[i] == 0:
        for j in range(n_feature):
            data_train[cnt1][j] = iris.data[i][j]
        data_train[cnt1][n_feature] = iris.target[i]
        cnt1 += 1
    else:
        for j in range(n_feature):
            data_test[cnt2][j] = iris.data[i][j]
        data_test[cnt2][n_feature] = iris.target[i]
        cnt2 += 1
# 初始化均值、方差以及类别比例
cnt = np.zeros((5, 5))
average = np.zeros((5, 5))
deviation = np.zeros((5, 5))
pro_feature = np.zeros(5)

# 将每个类别中的每个属性对应累加起来，average[i][j]代表所有标签为i的样本中第j个属性的总和
for i in range(n_target):
    for j in range(n_feature):
        for k in range(n_train):
            if data_train[k][n_feature] == i:
                average[i][j] += data_train[k][j]
                cnt[i][j] += 1.0

# 将上面求到的总和除以总数得到均值，average[i][j]代表所有标签为i的样本中第j个属性的均值
for i in range(n_target):
    for j in range(n_feature):
        average[i][j] /= cnt[i][j]

# 将每个类别中的每个属性减去均值的平方对应累加起来，deviation[i][j]代表所有标签为i的样本中第j个属性减去均值的平方和
for i in range(n_target):  # 差
    for j in range(n_feature):
        for k in range(n_train):
            if data_train[k][n_feature] == i:
                deviation[i][j] += (data_train[k][j] - average[i][j]) ** 2

# 将上面得到的综合除以总数，得到方差，deviation[i][j]代表所有标签为i的样本中第j个属性的方差
for i in range(n_target):
    for j in range(n_feature):
        deviation[i][j] /= cnt[i][j]

# 将每个类别的样本总数累加，pro_attr[i]代表每个类别的总数
for i in range(n_train):
    val = int(data_train[i][n_feature])
    pro_feature[val] += 1.0

# 将上面得到的类别总数除以训练样本数量得到比例，pro_attr[i]代表每个类别样本占总样本的比例
for i in range(n_target):
    pro_feature[i] /= n_train

# 根据公式对测试样本进行分类，并求准确度
n_correct = 0
for i in range(n_test):
    maxx = 0
    ans = 0
    for j in range(n_target):  # 求P(Cj|X)
        tmp = pro_feature[j]
        for k in range(n_feature):  # 求P(Xk|Cj)
            val = 1 / ((2 * math.pi) ** 0.5 * deviation[j][k] ** 0.5) * pow(e, -(
                        (data_test[i][k] - average[j][k]) ** 2) / (2.0 * deviation[j][k]))
            tmp *= val;
        if maxx < tmp:
            maxx = tmp
            ans = j
    if ans == data_test[i][n_feature]:
        n_correct += 1

print(n_correct / n_test)