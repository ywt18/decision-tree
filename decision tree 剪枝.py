import numpy as np
from sklearn.datasets import load_iris
import math
import operator
import random


# 计算数据熵
def calcShannonEnt(data):
    num_data = len(data)  # 获得数据长度
    labelCounts = {}  # 数据label集合
    for item in data:
        cu_label = item[-1]  # data最后一列是对应的label
        if cu_label not in labelCounts:
            labelCounts[cu_label] = 0  # 如果集合中没有，就加入一项新的label
        labelCounts[cu_label] += 1  # 将出现的label对应数量加一
    shannonEnt = 0
    for key in labelCounts:
        shannonEnt = shannonEnt - (labelCounts[key] / num_data) * math.log2(labelCounts[key] / num_data)  # 计算熵
    return shannonEnt


# 二分法对连续数据进行处理
def datato2(data):
    a_1 = list()
    a_2 = list()
    a_3 = list()
    a_4 = list()
    for item in data:
        a_1.append(item[0])
        a_2.append(item[1])
        a_3.append(item[2])
        a_4.append(item[3])
    # 排序
    s_1 = sorted(a_1)
    s_2 = sorted(a_2)
    s_3 = sorted(a_3)
    s_4 = sorted(a_4)
    # 计算分割点
    mid_1 = (s_1[74] + s_1[75]) / 2
    mid_2 = (s_2[74] + s_2[75]) / 2
    mid_3 = (s_3[74] + s_3[75]) / 2
    mid_4 = (s_4[74] + s_4[75]) / 2
    # 离散化
    for item in data:
        if (item[0] < mid_1):
            item[0] = 0
        else:
            item[0] = 1
        if (item[1] < mid_2):
            item[1] = 0
        else:
            item[1] = 1
        if (item[2] < mid_3):
            item[2] = 0
        else:
            item[2] = 1
        if (item[3] < mid_4):
            item[3] = 0
        else:
            item[3] = 1
    return data


# 三分法对连续数据进行处理
def datato3(data):
    a_1 = list()
    a_2 = list()
    a_3 = list()
    a_4 = list()
    for item in data:
        a_1.append(item[0])
        a_2.append(item[1])
        a_3.append(item[2])
        a_4.append(item[3])
    # 排序
    s_1 = sorted(a_1)
    s_2 = sorted(a_2)
    s_3 = sorted(a_3)
    s_4 = sorted(a_4)
    # 计算分割点
    mid_1_1 = (s_1[49] + s_1[50]) / 2
    mid_1_2 = (s_1[99] + s_1[100]) / 2
    mid_2_1 = (s_2[49] + s_2[50]) / 2
    mid_2_2 = (s_2[99] + s_2[100]) / 2
    mid_3_1 = (s_3[49] + s_3[50]) / 2
    mid_3_2 = (s_3[99] + s_3[100]) / 2
    mid_4_1 = (s_4[49] + s_4[50]) / 2
    mid_4_2 = (s_4[99] + s_4[100]) / 2
    # 离散化
    for item in data:
        if (item[0] < mid_1_1):
            item[0] = 0
        elif item[0] < mid_1_2:
            item[0] = 1
        else:
            item[0] = 2
        if (item[1] < mid_2_1):
            item[1] = 0
        elif item[1] < mid_2_2:
            item[1] = 1
        else:
            item[1] = 2
        if (item[2] < mid_3_1):
            item[2] = 0
        elif item[2] < mid_3_2:
            item[2] = 1
        else:
            item[2] = 2
        if (item[3] < mid_4_1):
            item[3] = 0
        elif item[3] < mid_4_2:
            item[3] = 1
        else:
            item[3] = 2
    return data


# 划分数据集
def splitData(data, axis, value):
    retData = []
    # 获得按照属性分类后的数据
    for item in data:
        if item[axis] == value:
            retData.append(item)
    return retData


# 选择最好的数据集划分方式
def chooseBestSplit(data):
    num_features = 4
    best_gain = 0
    best_feature = -1
    base_entropy = calcShannonEnt(data)  # 原始的熵
    # 遍历所有的属性计算每个属性的信息增益，找到信息增益最大的属性作为划分属性
    for i in range(num_features):
        all_value = [item[i] for item in data]
        all_value = set(all_value)
        new_entropy = 0
        # 遍历属性所有的值计算熵
        for value in all_value:
            split_data = splitData(data, i, value)
            new_entropy = new_entropy + len(split_data) / float(len(data)) * calcShannonEnt(split_data)
        # 计算信息增益
        new_gain = base_entropy - new_entropy
        print(new_gain)
        if new_gain > best_gain:
            best_gain = new_gain
            best_feature = i
    #设置阈值，判定是否进行划分
    if best_gain < 0.1:
        best_feature = -1
    return best_feature


# 叶节点属于同一类，则标为该类别，否则少数服从多数
def decide_class(classes):
    num_classes = {}
    # 找到类别最多的一类
    for item in classes:
        if item not in num_classes:
            num_classes[item] = 0  # 集合中没有就新增一项
        num_classes[item] += 1  # 对应的类别数量加一
        # 按照数量排序
    s_classes = sorted(num_classes.items(), key=operator.itemgetter(1), reverse=True)
    # 返回数量对多的类别
    return s_classes[0][0]


def is_same(data):
    temp = data[0]
    for i in range(1, len(data)):
        for j in range(len(temp)):
            if temp[j] != data[i][j]:
                return False
    return True


# 递归生成决策树
def creatTree(data, target):
    classes = [item[-1] for item in data]
    set_classes = set(classes)
    temp_target = target[:]
    print(classes)
    # 样本属于同一类
    if len(set_classes) == 1:
        print("类")
        print(classes[0])
        return classes[0]
    # 属性集为空或样本在属性集上取值相同，标记为样本数量最多的类
    elif len(target) == 0 or is_same(data):
        print("类")
        print(decide_class(classes))
        return decide_class(classes)
    # 从属性集中选取最优划分，递归建树
    else:
        print("划分")
        best_feature = chooseBestSplit(data)
        #剪枝，归为叶节点并将类别标记为样本数最多的类别
        if best_feature == -1:
            return decide_class(classes)
        print(best_feature)
        best_target = temp_target[best_feature]
        tree = {best_target: {}}
        feature_value = [item[best_feature] for item in data]
        feature_value = set(feature_value)
        # temp_target=np.delete(temp_target,best_feature)
        for item in feature_value:
            sub_target = temp_target[:]
            tree[best_target][item] = creatTree(splitData(data, best_feature, item), sub_target)
        return tree


# 对输入数据进行分类
def classify(tree, target, test_data):
    feature = list(tree.keys())[0]
    new_tree = tree[feature]
    index = target.index(feature)
    # 类似于建树的方式，递归获得分类结果
    res = 0
    for item in new_tree.keys():
        if item == test_data[index]:
            if type(new_tree[item]).__name__ == 'dict':
                res = classify(new_tree[item], target, test_data)
            else:
                res = new_tree[item]
    return res


if __name__ == "__main__":
    data = load_iris()  # 加载 IRIS 数据集
    print('keys: \n', data.keys())  # ['data', 'target', 'target_names', 'DESCR', 'feature_names']
    feature_names = data.get('feature_names')
    print('feature names: \n', data.get('feature_names'))  # 查看属性名称
    print('target names: \n', data.get('target_names'))  # 查看 label 名称
    x = data.get('data')  # 获取样本矩阵
    y = data.get('target')  # 获取与样本对应的 label 向量
    print(x.shape, y.shape)  # 查看样本数据
    all_data = np.insert(x, 4, values=y, axis=1)  # 将label加入到样本中，便于后续处理
    all_data = datato3(all_data)  # 对连续数据进行离散化处理
    print(all_data)
    # 生成训练样本和测试样本
    book = []
    test_data = []
    train_data = []
    for i in range(30):  # 每五个样本中，选一个作为测试样本
        val = 5 * i + random.randint(0, 4)
        test_data.append(all_data[val])
        book.append(val)
    for i in range(150):
        flag = 0
        for val in book:
            if i == val:
                flag = 1
                break
        if flag == 0:
            train_data.append(all_data[i])
    print("测试")
    print(len(test_data))
    print(len(train_data))
    tree = creatTree(train_data, data.get('feature_names'))  # 使用训练样本生成决策树

    # 对测试样本进行分类，输出准确率
    res = 0
    for item in test_data:
        print(item)
        print(classify(tree, data.get('feature_names'), item))
        if (classify(tree, data.get('feature_names'), item) == item[4]):
            res += 1
    print(res / 30)
    print(tree)
