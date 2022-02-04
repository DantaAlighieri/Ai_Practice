# 导入相应的库
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
# 随机生成样本数据。 二分类问题,每一个类别生成5000个样本数据
np.random.seed(12)
num_observations = 5000
x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observations),
               np.ones(num_observations)))
print (X.shape, y.shape)
# 数据的可视化
plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 1],
      c = y, alpha = .4)
plt.show()
from sklearn.linear_model import LogisticRegression
# 构建逻辑回归模型
# 两个参数同时去搜索 https://www.pythonheidong.com/blog/article/409633/8548b189d665a4467a66/
# 我们看solver参数，这个参数定义的是分类器，‘newton-cg’，‘sag’和‘lbfgs’等solvers仅支持‘L2’regularization，‘liblinear’ solver同时支持‘L1’、‘L2’regularization，若dual=Ture，则仅支持L2 penalty。
# 决定惩罚项选择的有2个参数：dual和solver，如果要选L1范数，dual必须是False，solver必须是liblinear
# 因此，我们只需将solver='liblinear’参数添加进去即可
logistic = linear_model.LogisticRegression(solver='liblinear')
# 惩罚项的类型,考虑两种
penalty = ['l1', 'l2']
# 惩罚项系数可能的取值,考虑10个不同的可能性
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
# 做交叉验证,5折交叉验证,利用已经设置好的hyperparameters
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# 开始训练,做完交叉验证之后,最好的模型存放在best_model中
best_model = clf.fit(X, y)
# 找到最好的超参数
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
