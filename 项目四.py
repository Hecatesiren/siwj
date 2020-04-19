import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


import math


class DecisionTreeClassifier(object):
    def __init__(self, criterion, degree):
        self.criterion = criterion
        self.order = []
        self.degree = degree
        self.delt = np.zeros([self.degree, x_train.shape[1]])
        self.rate = np.zeros([self.degree, x_train.shape[1]])
        self.feature = []
        self.classes_num = np.zeros([self.degree, self.degree, self.degree, self.degree])
        self.max = np.zeros(x_train.shape[1])
        self.min = np.zeros(x_train.shape[1])

    def info(self, y_train):
        Ent_D = 0
        for i in range(3):
            pk = np.sum(y_train == i) / y_train.shape[0]

            if (pk > 0):
                Ent_D = Ent_D - pk * math.log(pk, 2)
                print(Ent_D)
        return Ent_D

    # 信息增益
    def best_feature1(self, x_train, y_train):
        Ent_D = 0
        for i in range(3):  # 根节点信息熵
            pk = np.sum(y_train == i) / y_train.shape[0]
            if pk > 0:
                Ent_D = Ent_D - pk * math.log(pk, 2)
        gain = []
        for k in range(x_train.shape[1]):
            gain_var = 0
            for i in range(self.degree):
                self.rate[i, k] = np.sum(x_train[:, k] == i)
                ent = self.info(y_train[x_train[:, k] == i])  # 子节点信息熵
                gain_var = gain_var + self.rate[i, k] / y_train.shape[0] * ent
            gain.append(Ent_D - gain_var)  # 信息增益
            print(gain.append)

        gain_sort = sorted(gain)
        list = []
        for k in range(x_train.shape[1]):
            list.append(np.where(gain == gain_sort[x_train.shape[1] - 1 - k])[0][0])
        # pdb.set_trace()
        self.feature.append(list[0])
        return 0

    # 增益率
    def best_feature(self, x_train, y_train):
        Ent_D = 0
        for i in range(3):  # 根节点信息熵
            pk = np.sum(y_train == i) / y_train.shape[0]
            if pk > 0:
                Ent_D = Ent_D - pk * math.log(pk, 2)
        gain = []
        gain_ratio = []
        for k in range(x_train.shape[1]):
            gain_var = 0
            IV = 0
            for i in range(self.degree):
                self.rate[i, k] = np.sum(x_train[:, k] == i)
                ent = self.info(y_train[x_train[:, k] == i])  # 子节点信息熵
                IV = IV + self.rate[i, k] / y_train.shape[0]
                gain_var = gain_var + self.rate[i, k] / y_train.shape[0] * ent
            gain.append(Ent_D - gain_var)  # 信息增益
            gain_ratio.append((Ent_D - gain_var) / IV)  # 信息增益率
            print(gain_ratio.append)

        gain_average = np.average(gain)

        a = np.where(gain > gain_average)

        temp = 0
        idex = 0
        for i in a[0]:
            if gain_ratio[i] > temp:
                temp = gain_ratio[i]
                idex = i
        self.feature.append(idex)

        return 0

    from sklearn.externals.six import StringIO
    import pydot
    from sklearn import datasets
    from sklearn import tree

    iris = datasets.load_iris()
    clf = tree.DecisionTreeClassifier()
    X = iris.data
    y = iris.target

    clf.fit(X, y)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf("iris.pdf")

    # 基尼指数
    def best_feature3(self, x_train, y_train):
        Gini_index = []
        for k in range(x_train.shape[1]):
            gini_var = 0
            for i in range(self.degree):
                self.rate[i, k] = np.sum(x_train[:, k] == i)
                Gini = 1

                for j in range(3):  # 根节点信息熵
                    pk = np.sum(y_train[x_train[:, k] == i] == j) / y_train.shape[0]
                    Gini = Gini - pk * pk

                gini_var = gini_var + self.rate[i, k] / y_train.shape[0] * Gini
            Gini_index.append(gini_var)  # 信息增益

        a = np.where(Gini_index == np.min(Gini_index))
        self.feature.append(a[0][0])
        return 0

    def normal(self, x_train, flag=0):  # 将每一列数据划分为degree个类
        temp = np.zeros([x_train.shape[0], x_train.shape[1]])
        for k in range(x_train.shape[1]):
            if flag == 0:
                x1_max = max(x_train[:, k])
                x1_min = min(x_train[:, k])

                for j in range(self.degree):
                    self.delt[j, k] = x1_min + (x1_max - x1_min) / self.degree * j
            else:
                x1_max = self.max[k]
                x1_min = self.min[k]

            var = x_train[:, k].copy()

            for j in range(self.degree):  # 将每一列数据划分为degree个类
                var[x_train[:, k] >= self.delt[j, k]] = j

            temp[:, k] = var
            if (flag == 0):
                self.min[k] = x1_min
                self.max[k] = x1_max
        return temp

    def argmax(self, y_train):
        maxnum = 0
        MAX = 0
        for i in range(4):
            a = np.where(y_train == i)

            if (a[0].shape[0] > MAX):
                maxnum = i
                MAX = a[0].shape[0]
        return maxnum

    def getNum(self, y_train):
        MAX = 0
        maxnum = 0
        for i in range(y_train.shape[0]):
            a = np.where(y_train == i)
            if (a[0].shape[0] > MAX):
                MAX = a[0].shape[0]
                maxnum = i

        return MAX, maxnum

    def pre_pruning(self, x_var, y_var, c_num, depth=0):
        if depth == 4:
            return 0
        correct, id = self.getNum(y_var)
        cor = []
        for i in range(self.degree):
            a = np.where(x_var[:, self.feature[depth]] == i)
            cor.append(self.getNum(y_train[a])[0])
        if correct < np.sum(cor):
            for i in range(self.degree):
                a = np.where(x_var[:, self.feature[depth]] == i)
                self.pre_pruning(x_var[a], y_var[a], c_num[i], depth + 1)
        else:
            if depth == 0:
                c_num[:][:][:][:] = id
            elif depth == 1:
                c_num[:][:][:] = id
            elif depth == 2:
                c_num[:][:] = id
            elif depth == 3:
                c_num[:] = id
        return 0

    def pruning(self, x_var, y_var, c_num, depth=0):
        if depth == 4:
            return 0
        for i in range(self.degree):
            a = np.where(x_var[:, self.feature[depth]] == i)
            self.pre_pruning(x_var[a], y_var[a], c_num[i], depth + 1)
            correct, id = self.getNum(y_var)
            cor = []
            for i in range(self.degree):
                a = np.where(x_var[:, self.feature[depth]] == i)
                cor.append(self.getNum(y_train[a])[0])
            if correct > np.sum(cor):
                if depth == 0:
                    c_num[:][:][:][:] = id
                elif depth == 1:
                    c_num[:][:][:] = id
                elif depth == 2:
                    c_num[:][:] = id
                elif depth == 3:
                    c_num[:] = id
        return 0

    def fit(self, x_train, y_train):
        x_train = self.normal(x_train, flag=0)

        self.best_feature(x_train, y_train)
        for i in range(self.degree):
            a = np.where(x_train[:, self.feature[0]] == i)
            if len(a[0]):
                for j in range(self.degree):
                    self.best_feature(x_train[a], y_train[a])
                    b = np.where(x_train[a][:, self.feature[1]] == j)
                    if len(b[0]):
                        for k in range(self.degree):
                            self.best_feature(x_train[a][b], y_train[a][b])
                            c = np.where(x_train[a][b][:, self.feature[2]] == j)
                            if len(c[0]):
                                for p in range(self.degree):
                                    self.best_feature(x_train[a][b][c], y_train[a][b][c])
                                    d = np.where(x_train[a][b][c][:, self.feature[3]] == j)
                                    if len(d[0]):
                                        self.classes_num[i, j, k, p] = self.argmax(y_train[a][b][c][d])
                                    else:
                                        self.classes_num[i, j, k, p] = self.argmax(y_train[a][b][c])
                            else:
                                self.classes_num[i, j, k, :] = self.argmax(y_train[a][b])
                    else:
                        self.classes_num[i, j, :, :] = self.argmax(y_train[a])
            else:
                self.classes_num[i, :, :, :] = self.argmax(y_train)

        return 0

    def predict(self, x_test):
        y_show_hat = np.zeros([x_test.shape[0]])
        x_test = self.normal(x_test, 1)

        for j in range(x_test.shape[0]):
            var = int(x_test[j, self.feature[0]])
            var2 = int(x_test[j, self.feature[1]])
            var3 = int(x_test[j, self.feature[2]])
            var4 = int(x_test[j, self.feature[3]])
            y_show_hat[j] = self.classes_num[var, var2, var3, var4]
        # pdb.set_trace()
        return y_show_hat


iris_feature_E = ['sepal length', 'sepal width', 'petal length', 'petal width']
iris_feature = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    data = load_iris()
    x = data.data
    print(x.shape)
    y = data.target
    print(y.shape)

    list = [0]
    for i in range(8, 9):
        # x = x[:, :2]
        x_ALL, x_test, y_ALL, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=1)
        x_train, x_verify, y_train, y_verfy = train_test_split(x_ALL, y_ALL, train_size=0.75, test_size=0.25,
                                                               random_state=1)

        model = DecisionTreeClassifier(criterion='entropy', degree=i)
        model.fit(x_train, y_train)

        x_verify = model.normal(x_verify, flag=0)
        # model.pre_pruning(x_verify, y_verfy, model.classes_num)
        model.pruning(x_verify, y_verfy, model.classes_num)
        y_test_hat = model.predict(x_test)
        list.append(accuracy_score(y_test, y_test_hat))
        print(confusion_matrix(y_test, y_test_hat))
        print('postpruning accuracy_score:{:.2%}'.format(accuracy_score(y_test, y_test_hat)))

    plt.xlabel("degree")
    plt.ylabel("accuracy_score")
    plt.title("预剪枝")
    plt.plot(list)
    plt.show()

