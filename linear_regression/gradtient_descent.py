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


def gradient_descent(xVec, yVec):
    epsilon = 0.0005
    alpha = 0.0001
    X = mat(xVec)
    Y = mat(yVec).T
    theta = mat(ones(2)).T
    cost_funtion = []
    while True:
        # print(theta[0])
        J = 1 / (2 * len(X[0])) * (X.dot(theta) - Y).T * (X.dot(theta) - Y)
        print(J)
        cost_funtion.append(float(J))
        theta_new = theta - alpha * (X.T * (X * theta - Y))
        if (theta_new[0] - theta[0]) < epsilon and (
                theta_new[1] - theta[1]) < epsilon:
            return theta_new, cost_funtion
        theta = theta_new


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


def plot_cost_funciton():
    xVec, yVec = loadDataSet('ex0.txt')
    X = mat(xVec)
    Y = mat(yVec)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    j_cost = gradient_descent(xVec, yVec)[1]
    ax.scatter(range(0, len(j_cost)), j_cost)
    ax.plot(range(0, len(j_cost)), j_cost)
    plt.show()


plot_cost_funciton()
