import numpy as np
np.random.seed(12)
# 生成正负样本各100个
num_sample = 100
# 20乘20的矩阵
rand_m = np.random.rand(20, 20)
cov = np.matmul(rand_m.T, rand_m)
# 通过高斯分布生成样本
x1 = np.random.multivariate_normal(np.random.rand(20), cov, num_sample)
x2 = np.random.multivariate_normal(np.random.rand(20) + 5, cov, num_sample)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_sample), np.ones(num_sample)))
from sklearn.linear_model import LogisticRegression
# 使用L1的正则，C为控制正则的参数，C值越大，正则项的强度会越弱
clf = LogisticRegression(fit_intercept=True, C=0.1, penalty='l1', solver='liblinear')
clf.fit(X, y)
print("(L1)逻辑回归的参数w为", clf.coef_)
# 使用L2的正则，C为控制正则的参数，C值越大，正则项的强度就会越弱
clf = LogisticRegression(fit_intercept=True, C=0.1, penalty='l2', solver='liblinear')
clf.fit(X, y)
print("(L2)逻辑回归的参数w为", clf.coef_)
