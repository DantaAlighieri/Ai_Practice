# 导入包
import numpy as np
import pandas as pd

# 导入用于计数的包
from collections import Counter

# 导入tf-idf相关包
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# 导入模型评估的包
from sklearn import metrics

# 导入与word2vec相关的包
from gensim.models import KeyedVectors

# 导入与bert embedding相关的包，关于mxnet包下载的注意事项参考实验手册
from bert_embedding import BertEmbedding
import mxnet

# 包tqdm是用来对可迭代对象执行时生成一个进度条用的监视程序运行过程
from tqdm import tqdm

# 导入其他一些功能包
import requests
import os




