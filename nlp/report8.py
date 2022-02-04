import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# # read data from the dataset
# df = pd.read_csv("C:\\ai_workspace\dataset\dev.tsv", encoding='latin')
# # read 5 rows of the spam.csv
# context = df.head()
# print(context)
#
#
from pandas import DataFrame

from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import scipy as sp
import numpy as np

def read_file_test(path):
    student_data_to_load = os.path.join('Resources', path)
    test_df = pd.read_csv(student_data_to_load,
                          sep='\t',
                          lineterminator='\r',
                          quoting=3,
                          warn_bad_lines=True,
                          error_bad_lines=False,
                          skip_blank_lines=True).dropna()
    test_df.rename(columns={'Quality': 'Label',
                                '#1 ID': 'ID1',
                                '#2 ID': 'ID2',
                                '#1 String': 'Sentence1',
                                '#2 String': 'Sentence2'},
                                inplace=True)
    return test_df

def compute_similarity(path):

    data_df = pd.read_csv(path, sep='\t')
    data_df["cos"] = 0.0
    # print(data_df.head())

    training_labels = data_df['Quality']
    sentences1 = data_df['#1 String']
    sentences2 = data_df['#2 String']

    data_df["Quality"] = pd.to_numeric(data_df["Quality"])

    data_length = training_labels.size

    # print("data_length", data_length)

    # print(datetime.now().strftime("%H:%M:%S"), "creating modle")
    model = SentenceTransformer('stsb-roberta-large')
    #
    # print(datetime.now().strftime("%H:%M:%S"), "begin calculate")

    for i in range(data_length):
        s1 = sentences1[i]
        s2 = sentences2[i]

        embedding1 = model.encode(s1, convert_to_tensor=True)
        embedding2 = model.encode(s2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2).item()
        data_df["cos"][i] = cosine_scores

        # print(datetime.now().strftime("%H:%M:%S"), i, data_df["cos"][i])

    features1 = data_df["cos"]
    labels = data_df['Quality'].values

    rows, cols = (features1.size, 1)
    features = [[features1[j] for i in range(cols)] for j in range(rows)]

    # print(data_df.head(10))
    return labels, features

def read_file_test(path):
    student_data_to_load = os.path.join('Resources', path)
    data_df = pd.read_csv(student_data_to_load,
                          sep='\t',
                          lineterminator='\r',
                          quoting=3,
                          warn_bad_lines=True,
                          error_bad_lines=False,
                          skip_blank_lines=True).dropna()
    data_df.rename(columns={'Quality': 'Label',
                                '#1 ID': 'ID1',
                                '#2 ID': 'ID2',
                                '#1 String': 'Sentence1',
                                '#2 String': 'Sentence2'},
                                inplace=True)
    return data_df

def compute_test_similarity(path):

    test_df = read_file_test(path)

    test_test_df = test_df
    training_labels = test_test_df['Label']
    sentences1 = test_test_df['Sentence1']
    sentences2 = test_test_df['Sentence2']

    test_test_df["Label"] = pd.to_numeric(test_test_df["Label"])

    data_length = training_labels.size
    model = SentenceTransformer('stsb-roberta-large')
    test_test_df["cos"] = 0.0
    for i in range(data_length):
        s1 = sentences1[i]
        s2 = sentences2[i]

        embedding1 = model.encode(s1, convert_to_tensor=True)
        embedding2 = model.encode(s2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2).item()
        test_test_df[i]["cos"] = cosine_scores

    features1 = test_test_df["cos"]
    labels = test_test_df['Label'].values

    rows, cols = (features1.size, 1)
    features = [[features1[j] for i in range(cols)] for j in range(rows)]

    return labels, features

# 训练逻辑回归模型
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

training_labels, training_features = compute_similarity('C:\\ai_workspace\\dataset\\train.tsv')


# y1 = training_data_df.loc[:, training_data_df.columns == 'cos'].values.ravel()
# y2 = training_data_df.loc[:, training_data_df.columns == 'Quality'].values.ravel()

# print("=====================================================")
# print(y1.head())
# print(y2.head())
# print("=====================================================")

# tm = training_data_df.item()
# abc = tm["cos"]
# abc = tm["Quality"]



logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(training_features, training_labels)


test_labels, test_features = compute_test_similarity('C:\\ai_workspace\\dataset\\test.tsv')

y_pred = logreg.predict(test_features)
print(classification_report(test_labels, y_pred))




# data_df.to_csv("./result.tsv",
#                index=False,
#                sep='\t',
#                # escapechar='\\',
#                # doublequote=False,
#                )


# model = SentenceTransformer('stsb-roberta-large')
# embedding1 = model.encode(sentence1, convert_to_tensor=True)
# embedding2 = model.encode(sentence2, convert_to_tensor=True)
#
# cosine_scores1 = util.pytorch_cos_sim(embedding1, embedding2)
# print("Sentence 1:", sentence1)
# print("Sentence 2:", sentence2)
# print("Similarity score1,2:", cosine_scores1.item())




# df.head()
# df['numLabel'] = df['Label'].map({'ham': 0, 'spam': 1})
#
# print("ham email numbers:", len(df[df.numLabel == 0]), "spam email numbers:", len(df[df.numLabel==1]))
# print("totals:", len(df))








# # rename the title of the spam.csv so that it is easier to read
# # rename方法作用： 复制 DataFrame并对其索引index和列标签columns进行赋值。如果希望就地修改某个数据集，传入inplace=True即可
# # columns：dict-like or function，指定哪个列名，一般是字典形式，如：{'name':‘姓名’}，name是要替换的久列名，姓名是替换后的列名
# # inplace：bool, default False ：是否覆盖原来的数据
#
# df.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)
# df.head()
# df['numLabel'] = df['Label'].map({'ham': 0, 'spam': 1})
#
# print("ham email numbers:", len(df[df.numLabel == 0]), "spam email numbers:", len(df[df.numLabel==1]))
# print("totals:", len(df))
#
# # get the length of every email text content from 0 row to the end of the datasets
# text_lengths = [len(df.loc[i, 'Text']) for i in range(len(df))]
# print(text_lengths)
# # 设置图像，以text_lengths数组为数据，高度分段为100，蓝色，透明度0.5
# plt.hist(text_lengths, 100, facecolor='blue', alpha=0.5)
# # 设置x轴从0到200
# plt.xlim([0, 200])
# plt.show()


