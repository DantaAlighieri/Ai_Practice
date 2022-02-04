def edit_distance(str_1, str_2):
    # 初始化一个二维数组，用于记录单词转化的最小路径数
    len_str1 = len(str_1)
    len_str2 = len(str_2)
    # x轴表示目标字符串，y轴表示元字符串, 多加一个元素，是因为需要把0，0点的元素更新为0，方便计算
    distance_matrix = [[0 for i in range(len_str2 + 1)] for j in range(len_str1 + 1)]
    # 初始化第一行和第一列的距离

    # 遍历元字符串和目标字符串，用于计算距离
    for i in range(0, len_str1 + 1):
        for j in range(0, len_str2 + 1):
            if i == 0:
                distance_matrix[0][j] = j
            if j == 0:
                distance_matrix[i][0] = i
            if i != 0 and j != 0:
                if str_1[i - 1] == str_2[j - 1]:
                    # 当两个字符相等的时候，距离不会增加
                    distance_matrix[i][j] = min(distance_matrix[i - 1][j] + 1, distance_matrix[i][j - 1] + 1,
                                                distance_matrix[i - 1][j - 1])
                else:
                    # 当两个字符不相等的时候，距离增加1
                    distance_matrix[i][j] = min(distance_matrix[i - 1][j] + 1, distance_matrix[i][j - 1] + 1,
                                                distance_matrix[i - 1][j - 1] + 1)
    return distance_matrix[len_str1][len_str2]


str1 = "hello"
str2 = "hello world"
print("%s变成%s需要%d步" % (str1, str2, edit_distance(str1, str2)))