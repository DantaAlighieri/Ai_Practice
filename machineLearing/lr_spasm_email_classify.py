import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# read data from the dataset
df = pd.read_csv("C:\\ai_workspace\dataset\spam.csv", encoding='latin')
# read 5 rows of the spam.csv
context = df.head()
print(context)
# rename the title of the spam.csv so that it is easier to read
# rename方法作用： 复制 DataFrame并对其索引index和列标签columns进行赋值。如果希望就地修改某个数据集，传入inplace=True即可
# columns：dict-like or function，指定哪个列名，一般是字典形式，如：{'name':‘姓名’}，name是要替换的久列名，姓名是替换后的列名
# inplace：bool, default False ：是否覆盖原来的数据

df.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)
df.head()
df['numLabel'] = df['Label'].map({'ham': 0, 'spam': 1})

print("ham email numbers:", len(df[df.numLabel == 0]), "spam email numbers:", len(df[df.numLabel==1]))
print("totals:", len(df))

# get the length of every email text content from 0 row to the end of the datasets
text_lengths = [len(df.loc[i, 'Text']) for i in range(len(df))]
print(text_lengths)
# 设置图像，以text_lengths数组为数据，高度分段为100，蓝色，透明度0.5
plt.hist(text_lengths, 100, facecolor='blue', alpha=0.5)
# 设置x轴从0到200
plt.xlim([0, 200])
plt.show()

# 导入库，用来把文本转化为向量形式
from sklearn.feature_extraction.text import CountVectorizer
# 构建文本的向量，基于词频的表示
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.Text)
y = df.numLabel
# 把数据分成测试集和训练集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
print("训练数据中的样本个数：", X_train.shape[0], "测试数据中的样本个数：", X_test.shape[0])
# 利用朴素贝叶斯做训练
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy on test data:", accuracy_score(y_test, y_pred))
# print 混淆矩阵
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(confusion_matrix)



