import pandas as pd

# 构建数据集包


df = pd.DataFrame({
    'wing': [12, 10, 13, 10, 13, 12, 15, 12],
    'body': [15, 20, 23, 27, 30, 36, 39, 42],
    'type': [0, 0, 0, 0, 1, 1, 1, 1]
})

print(df.head(1))
print(df.tail(8))

print("===============================================================")

import matplotlib.pyplot as plt

# show previous data(df) as Implot
# Implot is a 2D scatterplot
plt.scatter(df.loc[:, 'wing'], df.loc[:, 'body'], c=df.loc[:, 'type']
            , cmap=plt.cm.cool)

plt.legend()
plt.show()

from sklearn import datasets

iris = datasets.load_iris()

df2 = pd.DataFrame(iris.data, columns=iris.feature_names)
df2['target'] = iris.target
print(df2.head())

import seaborn as sns

sns.pairplot(df2, hue="target")

dat = datasets.make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=1, n_classes=2)
df3 = pd.DataFrame(dat[0], columns=['var1', 'var2', 'var3'])
df3['class'] = dat[1]
sns.stripplot(data=df3)




