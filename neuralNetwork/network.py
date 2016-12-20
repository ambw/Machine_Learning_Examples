import numpy as np
from mnist import MNIST
import pandas as pd
import time


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def nnCostFunction(theta1, theta2, input_layer_size, hidden_layer_size,
                   num_labels, X, y_matrix, lamada):
    # 训练集的个数
    m = X.shape[0]
    # cost
    a1 = np.c_[np.ones((m, 1)), X]  # m x input_layer_size + 1
    # m x input_layer_size+1 * input_layer_size+1 x hidden_layer_size
    z2 = a1.dot(theta1.T)
    # m x (hidden_layer_size+1)
    a2 = np.c_[np.ones((m, 1)), sigmoid(z2)]
    # m x hidden_layer_size + 1 * hidden_layer_size+ 1 x num_labels
    z3 = a2.dot(theta2.T)
    # m x num_labels
    a3 = sigmoid(z3)

    J = (-1 / m) * np.sum(
        (np.log(a3) * y_matrix + np.log(1 - a3) * (1 - y_matrix)))
    # reggularized
    reg = (lamada / (2 * m)) * \
          (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))

    # regularized cost function
    J = J + reg

    # Gradients
    delta3 = a3 - y_matrix  # m x num_labels
    # hide_layer_size  x m
    delta2 = theta2[:, 1:].T.dot(delta3.T) * (sigmoidGradient(z2).T)
    # lables x hidden_layer_size + 1
    Delta3 = delta3.T.dot(a2)
    # hidden_layer_size x input_layer_size + 1
    Delta2 = delta2.dot(a1)

    theta1_ = np.c_[np.zeros((theta1.shape[0], 1)), theta1[:, 1:]]
    theta2_ = np.c_[np.zeros((theta2.shape[0], 1)), theta2[:, 1:]]

    theta1_grad = Delta2 / m + (theta1_ * lamada) / m
    theta2_grad = Delta3 / m + (theta2_ * lamada) / m
    return (J, theta1_grad, theta2_grad)


def gradientDescent(X, y, theta1, theta2, input_layer_size, hidden_layer_size,
                    num_labels, lamada, num_iters, alpha):
    for i in range(num_iters):
        J, theta1_grad, theta2_grad = nnCostFunction(
            theta1, theta2, input_layer_size, hidden_layer_size, num_labels, X,
            y, lamada)
        theta1 -= alpha * theta1_grad
        theta2 -= alpha * theta2_grad
        if (i % 10 == 0):
            print(i + 1, J)
    return theta1, theta2


def loadMINST():
    mndata = MNIST('./data')
    # load traning set
    training_imgs_list, training_labels_list = mndata.load_training()
    # load test set
    test_imgs_list, test_labels_list = mndata.load_testing()
    # turn list to numpy.array
    training_imgs = np.asarray(training_imgs_list)
    training_labels = np.asarray(training_labels_list)

    test_imgs = np.asarray(test_imgs_list)
    test_labels = np.asarray(test_labels_list)

    return (training_imgs, training_labels, test_imgs, test_labels)


def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))  # 对应theta的权重
    epsilon_init = (6.0 / (L_out + L_in))**0.5
    W = np.random.rand(
        L_out, 1 + L_in
    ) * 2 * epsilon_init - epsilon_init  # np.random.rand(L_out,1+L_in)产生L_out*(1+L_in)大小的随机矩阵
    return W


def predict(theta1, theta2, X):
    m = X.shape[0]
    num_labels = theta2.shape[0]
    # 正向预测结果

    a1 = np.c_[np.ones((m, 1)), X]  # m x input_layer_size + 1
    # m x input_layer_size+1 * input_layer_size+1 x hidden_layer_size
    z2 = a1.dot(theta1.T)
    # m x (hidden_layer_size+1)
    a2 = np.c_[np.ones((m, 1)), sigmoid(z2)]
    # m x hidden_layer_size + 1 * hidden_layer_size+ 1 x num_labels
    z3 = a2.dot(theta2.T)
    # m x num_labels
    a3 = sigmoid(z3)
    p = np.array(np.where(a3[0, :] == np.max(a3, axis=1)[0]))
    for i in np.arange(1, m):
        t = np.array(np.where(a3[i, :] == np.max(a3, axis=1)[i]))
        p = np.vstack((p, t))
    return p


def test():
    training_imgs, training_labels, test_imgs, test_labels = loadMINST()
    input_layer_size = training_imgs.shape[1]
    hidden_layer_size = 25
    num_labels = 10
    theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    # 处理y 把y 从mx1的标签 变为 m x num_labels
    training_labels_matrix = pd.get_dummies(training_labels.ravel()).as_matrix(
    )
    # 处理y 把y 从mx1的标签 变为 m x num_label
    test_labels_matrix = pd.get_dummies(test_labels.ravel()).as_matrix()

    start = time.time()
    print('start time:',
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))
    theta1, theta2 = gradientDescent(
        training_imgs, training_labels_matrix, theta1, theta2,
        input_layer_size, hidden_layer_size, num_labels, 1, 1500, 0.03)
    print('执行时间：', time.time() - start)
    print('当前时间:',
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    p = predict(theta1, theta2, test_imgs)
    right_sum = 0.0
    for i in range(test_imgs.shape[0]):
        if p[i] == test_labels[i]:
            right_sum += 1.0
        print('准确率：', 100 * right_sum / (i + 1), '\%')
    print('准确率：', 100 * right_sum / test_imgs.shape[0], '\%')
    print('当前时间:',
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('执行时间：', time.time() - start)


if __name__ == "__main__":
    test()
