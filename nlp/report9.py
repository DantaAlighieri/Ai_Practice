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

    training_labels = data_df['Quality']
    sentences1 = data_df['#1 String']
    sentences2 = data_df['#2 String']

    data_df["Quality"] = pd.to_numeric(data_df["Quality"])

    data_length = training_labels.size


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
                          quoting=3).dropna()
    data_df.rename(columns={'Quality': 'Label',
                                '#1 ID': 'ID1',
                                '#2 ID': 'ID2',
                                '#1 String': 'Sentence1',
                                '#2 String': 'Sentence2'},
                                inplace=True)
    return data_df

def compute_test_similarity(path):

    test_test_df = read_file_test(path)
    training_labels = test_test_df['Label']
    sentences1 = test_test_df['Sentence1']
    sentences2 = test_test_df['Sentence2']
    data_length = training_labels.size
    model = SentenceTransformer('stsb-roberta-large')
    test_test_df["cos"] = 0.0
    # cos_array = np.zeros(data_length)
    # cosine_scores = 0.0
    for i in range(data_length):
        if i in sentences1.keys():
            s1 = sentences1[i]
            s2 = sentences2[i]
            embedding1 = model.encode(s1, convert_to_tensor=True)
            embedding2 = model.encode(s2, convert_to_tensor=True)
            # cosine_scores = round(util.pytorch_cos_sim(embedding1, embedding2).item(), 1)
            test_test_df["cos"] = util.pytorch_cos_sim(embedding1, embedding2).item()

    # test_test_df["cos"] = cos_array
    features1 = test_test_df["cos"]
    labels = test_test_df['Label'].values

    rows, cols = (features1.size, 1)

    # for num in features1:
    #     [num for i in range(cols)] for j in range(rows)]

    features = [[features1[j] for i in range(cols)] for j in range(rows)]

    return labels, features

# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

training_labels, training_features = compute_similarity('C:\\ai_workspace\\dataset\\train.tsv')

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(training_features, training_labels)

test_labels, test_features = compute_test_similarity('C:\\ai_workspace\\dataset\\test.tsv')

y_pred = logreg.predict(test_features)
print(classification_report(test_labels, y_pred))
