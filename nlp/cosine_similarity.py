import numpy as np


def cosine_similarity(v1, v2):
    # 计算2个向量的余弦相似度
    # 计算内积
    dot_product = np.dot(v1, v2)
    # 计算v1，v2的长度
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v1)
    return dot_product / (norm_v1 * norm_v2)


sentence_v1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
sentence_v2 = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0])
sentence_v3 = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1])

print(sentence_v1, "和", sentence_v2, "的相似度为%s" %(cosine_similarity(sentence_v1, sentence_v2)))
print(sentence_v1, "和", sentence_v3, "的相似度为%s" %(cosine_similarity(sentence_v1, sentence_v3)))
