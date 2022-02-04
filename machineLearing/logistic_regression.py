import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 随机生成样本数据，正类和负类各5000个样本数据
np.random.seed(12)
num_sample = 5000
# x1, x2为5000行2列的数组， 2维， x1，均值u在x轴和y轴的为0，0点，方差是否可以理解为在x轴和y轴分别为以点[1, 0.75], [0.75, 1]附近
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_sample)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_sample)

# x1, x2 分别为5000行2列，可以理解为有5000个x，y轴上的点，垂直合并以后的数组x为10000行2列，即10000个x，y轴上的点
x = np.vstack((x1, x2)).astype(np.float32)
# y1为1行5000列，y2为1行5000列，水平合并以后，y为10000行1列的数组，其中0有5000，1有5000
y1 = np.zeros(num_sample)
y2 = np.ones(num_sample)
y = np.hstack((y1, y2))

# 画图
plt.figure(figsize=(12, 8))
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=.4)
plt.show()


# sigmoid関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 目标函数最大化
def maxTargetFunction(x, y, w, b):
    # 获取正样本和负样本的位置
    positive, negative = np.where(y == 1), np.where(y == 0)
    # 计算正样本的目标函数值
    positive_max_value = np.sum(np.log(sigmoid(np.dot(x[positive], w) + b)))
    # 计算负样本的目标函数值
    negative_max_value = np.sum(np.log(1 - sigmoid(np.dot(x[negative], w) + b)))
    # 求最小值，两者相加
    return -(positive_max_value + negative_max_value)


# 逻辑回归 随机梯度下降法
def logistic_regression_minibatch(x, y, steps, learning_rate):
    """
    基于梯度下降法实现逻辑回归
    steps:迭代次数
    learning_rate 学习率，即步长
    """
    # 初始化w, b shap[0]表示矩阵的行数, shape[1]表示矩阵的列数
    # w [0, 0]
    w = np.zeros(x.shape[1])
    b = 0
    for step in range(steps):
        # 随机抽取一个batch，进行梯度计算, 基于单个样本的梯度进行参数的更新
        # 随机从a中以概率p选取3个，p没有指定的时候，相当于是一致的分布
        # np.random.choice(a = 5, size = 3, replace = False, p = None)
        batch = np.random.choice(x.shape[0], 100)
        x_batch, y_batch = x[batch], y[batch]
        # 计算差值 np.dot计算单个数组的内积或者两个矩阵之间的矩阵积
        difference = sigmoid(np.dot(x_batch, w) + b) - y_batch
        # 对w和t参数进行梯度更新
        w = w - learning_rate * (np.matmul(x_batch.T, difference))
        b = b - learning_rate * (np.sum(difference))
        # 每隔1000点打印一次，w和b的返回值，观察是否有变化
        if step % 100000 == 0:
            print(maxTargetFunction(x, y, w, b))
    return w, b


w, b = logistic_regression_minibatch(x, y, steps=5000000, learning_rate=5e-4)
print("逻辑回归实现，w，b参数分别是：", w, b)

# 调用sklearn模型训练
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=True, C=5e-4)
clf.fit(x, y)
print("sklearn逻辑回归的参数w，b分别为：", clf.coef_, clf.intercept_, )
