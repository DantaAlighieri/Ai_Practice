import pandas as pd
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

from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

student_data_to_load = os.path.join('Resources', 'C:\\ai_workspace\dataset\dev.tsv')
student_data_df = pd.read_csv(student_data_to_load, sep='\t', warn_bad_lines=True, error_bad_lines=False)

student_data_df.rename(columns={'Quality': 'Label',
                                '#1 ID': 'ID1',
                                '#2 ID': 'ID2',
                                '#1 String': 'Sentence1',
                                '#2 String': 'Sentence2'},
                       inplace=True)
student_data_df["cos"] = 0
student_data_df.head()

labels = student_data_df['Label']
sentences1 = student_data_df['Sentence1']
sentences2 = student_data_df['Sentence2']
data_length = labels.size
model = SentenceTransformer('stsb-roberta-large')
for i in range(data_length):
    embedding1 = model.encode(sentences1[i], convert_to_tensor=True)
    embedding2 = model.encode(sentences2[i], convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    print(cosine_scores.item())
    # print(cosine_scores.)

    student_data_df["cos"][i] = cosine_scores.item()

print(student_data_df.head())
print(student_data_df.head())



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


