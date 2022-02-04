# 通过线性回归模型拟合身高，体重。然后对新的数据进行预测

# 引用sklearn库，其中包含线性回归模块
from sklearn import datasets, linear_model

# 把数据拆分成训练集和测试集
from sklearn.model_selection import train_test_split

# 引用 numpy库，做科学计算
import numpy as np

# 引用matplotlib库，主要用来画图
import matplotlib.pyplot as plt

# 创建数据集，并将其写入np数组中 np array和python list的区别如下：
# python 中的 list 是 python 的内置数据类型，list 中的数据类型不必相同，
# 在 list 中保存的是数据的存放的地址，即指针，并非数据。
# array() 是 numpy 包中的一个函数，array 里的元素都是同一类型。
data = np.array([[152, 51], [156, 53], [160, 54], [164, 55],
        [168, 57], [172, 60], [176, 62], [180, 65],
        [184, 69], [188, 72]])
print("数据大小(%d,%d)：" %data.shape)

# 将数据存放在X特征向量和y标签中。因为后面调用模型的时候对特征向量有要求，所以需要把X，y转化为矩阵形式
X, y = data[:, 0].reshape(-1, 1), data[:, 1]

# 使用train_test_split函数把数据随机分为训练数据和测试数据。训练数据的占比由参数train_size决定
# 例如：当train_size=0.8时，表示80%的数据将会作为训练数据，20%的数据为测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# 创建线性回归模型
regr = linear_model.LinearRegression()

# 在X_train, y_train上训练线性回归模型。训练结束后，结果模型将会存储到regr中
regr.fit(X_train, y_train)

# 计算训练好的模型在训练数据上得拟合程度 R:决定系数
print("训练数据决定稀疏R: %.2f" % regr.score(X_train, y_train))

# 画训练数据图
plt.scatter(X_train, y_train, color='black')
# 画出训练数据上已经拟合完毕的直线
plt.plot(X_train, regr.predict(X_train), color='blue')

# 画出测试数据
plt.scatter(X_test, y_test, color='pink')

# 添加x，y轴标签
plt.xlabel('height(cm)')
plt.ylabel('weight(kg)')
plt.show()

# 输出在测试数据上的决定系数
print('测试数据决定系数：%.2f' % regr.score(X_test, y_test))
print("测试模型，给出身高，输出体重: %.2f" % regr.predict([[199]]))

print("训练数据特征：", X_train, "训练数据标签", y_train)
print("测试数据特征：", X_test, "测试数据标签", y_test)









