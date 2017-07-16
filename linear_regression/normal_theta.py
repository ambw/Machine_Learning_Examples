import matplotlib.pyplot as plt
from numpy import *


def loadDataSet(filename):
    file_stream = open(filename)
    num_feature = len(file_stream.readline().split('\t')) - 1
    data_matrix = []
    label_matrix = []
    for line in file_stream.readlines():
        line_array = []
        # strip remove leading and tail char
        current_line = line.strip().split("\t")
        for i in range(num_feature):
            line_array.append(float(current_line[i]))
        data_matrix.append(line_array)
        label_matrix.append(float(current_line[-1]))
    return data_matrix, label_matrix


def normalE_theta(xVec, yVec):
    X = mat(xVec)
    Y = mat(yVec).T
    XTX = X.T * X
    if linalg.det(XTX) == 0:  # 计算行列式,如果为奇异矩阵则不能求逆矩阵
        print('This is a singular matrix, it canot be inverse')
        return
    theta = XTX.I * X.T * Y
    return theta


def plotPoint(linear_regress_funciton):
    xVec, yVec = loadDataSet('ex0.txt')
    X = mat(xVec)
    Y = mat(yVec)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 1].flatten().A[0], Y.T[:, 0].flatten().A[0])
    xCopy = X.copy()
    xCopy.sort(0)  # sort(0) 中的0指定排序参数
    theta = linear_regress_funciton(xVec, yVec)[0]
    print(theta)
    yHat = xCopy * theta
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


plotPoint(normalE_theta)