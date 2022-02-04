
## 员工离职率预测小案例
# 引入相应的工具包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
from sklearn.model_selection import train_test_split
# 读取数据到pandas dataframe

df = pd.read_csv("C:\\ai_workspace\dataset/HR_comma_sep.csv", index_col=None)
# 检测是否有缺失数据 有true，没有false，isnull，将缺失数据变为true，未缺失变为false
print (df.isnull().any(), "\n\n")
# 看看数据的样例吧
print (df.head(), "\n\n")


# # 给定数据里的列名有些不太清楚,咱们改改吧!
df = df.rename(columns={'satisfaction_level': 'satisfaction',
            'last_evaluation': 'evaluation',
            'number_project': 'projectCount',
            'average_montly_hours': 'averageMonthlyHours',
            'time_spend_company': 'yearsAtCompany',
            'Work_accident': 'workAccident',
            'promotion_last_5years': 'promotion',
            'sales' : 'department',
            'left' : 'turnover'
            })

# # 将string类型转换为整数类型,不然后面处理不了。
df["department"] = df["department"].astype('category').cat.codes
df["salary"] = df["salary"].astype('category').cat.codes

# 类别型变量需要转换成独热编码形式，列出所有类别型变量
cat_vars = ['satisfaction', 'evaluation', 'projectCount', 'averageMonthlyHours', 'yearsAtCompany', 'workAccident', 'promotion', 'department', 'turnover']
for var in cat_vars:
    # TODO 每个变量转换成类别型变量， 参考函数 pd.get_dummies
    cat_list = pd.get_dummies(df[var], prefix=var)
    df = df.join(cat_list)

# 剔除掉原来类别型变量，只保留独热编码
data_final = df.drop(cat_vars, axis=1)
data_final.columns.values
print(data_final)


# X = data_final.drop('turnover', axis=1)
# y = data_final['turnover']

# TODO 构造训练数据，X为特征，y为标签
X = data_final.loc[:, data_final.columns != 'turnover']
y = data_final.loc[:, data_final.columns == 'turnover'].values.ravel()

# TODO 把数据分为训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123, stratify=y)

# 准备工作就绪,接下来就训练模型时间到了!
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# 决训练一下决策树
dtree = tree.DecisionTreeClassifier(
  criterion='entropy',
  #max_depth=3, # 定义树的深度, 可以用来防止过拟合
  min_weight_fraction_leaf=0.01 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
  )
dtree = dtree.fit(X_train,y_train)
print ("\n\n ---决策树---")
print(classification_report(y_test, dtree.predict(X_test)))
# 随机森林
rf = RandomForestClassifier(
  criterion='entropy',
  n_estimators=1000,
  max_depth=None, # 定义树的深度, 可以用来防止过拟合
  min_samples_split=10, # 定义至少多少个样本的情况下才继续分叉
  #min_weight_fraction_leaf=0.02 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
  )
rf.fit(X_train, y_train)
print ("\n\n ---随机森林---")
print(classification_report(y_test, rf.predict(X_test)))

print ("\n\n ---逻辑回归---")
# 训练逻辑回归模型
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# TODO 初始化逻辑回归模型并在训练数据上训练
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)
# TODO 计算F1-SCORE， 使用classification_report函数
y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))

