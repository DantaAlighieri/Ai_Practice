import jieba

# 基于jieba的分词，结巴词库不包括"北陆先端科学技术大学院大学" 关键词
text_content = "北陆先端科学技术大学院大学，简称JAIST，是1990年日本设立的研究院性质国立大学。北陆先端科学技术大学院大学的建学目的是创造出世界最高水准的丰富的学习环境，培养能够在下一个时代担任科学技术创造的指导性人才，建造世界级最高水准的高等教育研究机构。"
set_list = jieba.cut(text_content, cut_all=False)
print("Default Mode:" + "/".join(set_list))
# 添加关键词
jieba.add_word("北陆先端科学技术大学院大学")
set_list = jieba.cut(text_content, cut_all=False)
print("Default Mode: " + "/".join(set_list))

dic = set(["北陆先端科学技术大学院大学", "建学目的", "创造出", "世界", "最高水准", "丰富", "学习环境"])


# def word_break(str):
#     jieba.load_userdict(dic)
#     word_list = jieba.cut(str, cut_all=False)
#     print(word_list)
#     word_list_length = len(word_list)
#     if len(word_list_length) == 1:
#         return True
#     else:
#         return False
#
# word_break("北陆先端科学技术大学院大学的建学目的")
#
# assert word_break("北陆先端科学技术大学院大学的建学目的") == True
# assert word_break("世界最高水准") == True
# assert word_break("丰富的学习环境") == False
# assert word_break("世界最高水准的丰富的学习环境") == False


# 构建停用词词典
stop_words = ["the", "an", "is", "there"]
# 假设word_list包含了文本里的单词
word_list = ["we", "are", "the", "students"]
filtered_words = [word for word in word_list if word not in stop_words]
print(filtered_words)
# 调包
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# cachedStopWords = stopwords.words("the")
#
# word_tokens = word_tokenize(word_list)
# print(word_tokens)
# print(filtered_words)


from nltk.stem.porter import *
stemmer = PorterStemmer()
test_strs = ['caresses', 'flies', 'dies', 'mules', 'denied',
    'died', 'agreed', 'owned', 'humbled', 'sized',
    'meeting', 'stating', 'siezing', 'itemization',
    'sensational', 'traditional', 'reference', 'colonizer',
    'plotted']
singles = [stemmer.stem(word) for word in test_strs]
print(' '.join(singles))

