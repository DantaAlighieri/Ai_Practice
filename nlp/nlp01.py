import matplotlib as matplotlib  # this is used as mathmatic package
import notebook as notebook  # graphic package
import numpy as np
# Get the interactive Tools for Matplotlib

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

from sklearn.manifold import TSNE  # word similarity package
from sklearn.decomposition import PCA

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('C:ai_work_space\Corpus\glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

word2List = model.most_similar('banana')
for wordProbability in word2List:
    print(wordProbability)

word2NegList = model.most_similar(negative='banana')
for word2NegProbability in word2NegList:
    print(word2NegProbability)

# analogy test
result = model.most_similar(positive=['woman', 'king'], negative=['man'])

print("{}:{:.4f}".format(*result[0]))


def analogy(x1, x2, y1):
    analogyResult = model.most_similar(positive=[y1, x2], negative=[x1])
    return analogyResult[0][0]


print(analogy('japan', 'japanese', 'china'))
