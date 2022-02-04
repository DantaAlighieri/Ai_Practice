from nltk.tokenize import word_tokenize
import numpy as np
import math


def read_file(doc_list):
    file_map = {}
    for doc in doc_list:
        f = open("C:\\ai_workspace\dataset/" + doc, "r")
        lines = f.readlines()
        file_map[doc] = lines
    return file_map


def build_corpus(file_map):
    corpus = []
    for lines in file_map.values():
        for line in lines:
            if line.strip():
                # tokens = word_tokenize(line, "english")
                tokens = line.split(" ");
                for token in tokens:
                    if token not in corpus:
                        corpus.append(token)
    return corpus


def build_incidence_matrix(m, n, corpus, doc_list, file_map):
    incidence_matrix = np.zeros((m, n))
    for doc in file_map.keys():
        lines = file_map[doc]
        for line in lines:
            if line.strip():
                tokens = line.split(" ");
                for token in tokens:
                    incidence_matrix[corpus.index(token)][doc_list.index(doc)] = incidence_matrix[corpus.index(token)][
                                                                                     doc_list.index(doc)] + 1
    return incidence_matrix


# to compute how many document that a word occurs
def build_incidence_in_corpus(m, n, corpus, incidence_matrix):
    incidence_in_corpus = {}
    for row in range(m):
        count = 0
        for column in range(n):
            if incidence_matrix[row][column] > 0:
                count = count + 1
        incidence_in_corpus[corpus[row]] = count
    return incidence_in_corpus


def build_tf_idf_matrix(M, N, corpus, incidence_matrix, incidence_in_corpus):
    tf_idf_matrix = np.zeros((M, N))
    for doc in range(N):
        for row in range(M):
            tf = incidence_matrix[row][doc]
            idf = math.log(N / incidence_in_corpus[corpus[row]],2)
            tf_idf_matrix[row][doc] = '%.2f' % (tf * idf)
    return tf_idf_matrix


def tf_idf(doc_list):
    file_map = read_file(doc_list)
    corpus = build_corpus(file_map)
    N = len(doc_list)
    M = len(corpus)
    incidence_matrix = build_incidence_matrix(M, N, corpus, doc_list, file_map)
    incidence_in_corpus = build_incidence_in_corpus(M, N, corpus, incidence_matrix)
    tf_idf_matrix = build_tf_idf_matrix(M, N, corpus, incidence_matrix, incidence_in_corpus)
    return tf_idf_matrix


docs = ["D1.txt", "D2.txt", "D3.txt", "D4.txt"]
# A = tf_idf(docs)
A = tf_idf(docs)

print(A)

