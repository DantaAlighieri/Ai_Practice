# 导入词向量的包
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "泽兰逢春茂盛芳馨，桂花遇秋皎洁清新。",
    "兰桂欣欣生机勃发，春秋自成佳节良辰。",
    "谁能领悟山中隐士，闻香深生仰慕之情？",
    "花卉流香原为天性，何求美人采撷扬名。"
]
corpus_2 = [
    "兰花到了春天枝叶茂盛，桂花遇秋天则皎洁清新。兰桂欣欣向荣生机勃发，所以春秋也成了佳节良辰。可是谁能领悟山中隐士，见到此情此景而产生的仰慕之情？花木流香原为天性，它们并不求美人采撷扬名。"
]

# 设置停用词，过滤掉标点符号
stop_words = ["，", "。", "？"]
# 创建分词列表集合
split_list = []
# 导入jieba分词，对语料库进行分词

import jieba

for i in range(len(corpus)):
    set_list = jieba.cut(corpus[i], cut_all=False)
    split_list = split_list + ",".join(set_list).split(",")

# split_list = 过滤掉标点符号
filtered_words = [word for word in split_list if word not in stop_words]

# 构建CountVectorizer对象
vectorizer = CountVectorizer()
# 生成文档的count vector
X = vectorizer.fit_transform(filtered_words)
# 打印词典库
print(vectorizer.get_feature_names())
# 打印每个文档的向量
print(X.toarray())