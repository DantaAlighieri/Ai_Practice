def generate_edit_one(str):
    """
    给定一个字符串,生成编辑距离为1的字符串列表。
    个人觉得处理方式有点函数式编程的思维的感觉，类似于haskell的书写方法
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    # 假设单词为apple
    # word[:1]从0取到下角标第1位 = a， word[:2]从0取到下角标第2位= ap
    # word[1:]从下角标第1位开始取, 取到单词末尾 = pple, word[2:]从下角标第2位开始取，取到末尾 = pple
    # 创建二维数组 splits [('', 'apple'), ('a', 'pple'), ('ap', 'ple'), ('app', 'le'), ('appl', 'e'), ('apple', '')]
    splits = [(str[:i], str[i:]) for i in range(len(str) + 1)]
    # inserts 代码解释：
    # 将splits数组分为左L和右R两部分，将letters数组进行切割，每个的单词元素为c，inserts数组为，splits数组左 加 letters每一个字母 加 splits数组右构成的数组
    # 例： splits第一个元素：('', 'apple') --> [''(左) + a + 'apple'(右), ''(左) + b + 'apple'(右), ... ''(左) + z + 'apple'(右)]
    # 例： splits第二个元素：('a', 'pple') --> ['a'(左) + a + 'pple'(右), 'a'(左) + b + 'pple'(右), ... 'a'(左) + z + 'pple'(右)]
    # ...
    # 例： splits最后个元素：('apple', '') --> ['apple'(左) + a + ''(右), 'apple'(左) + b + ''(右), ... 'apple'(左) + z + ''(右)]
    inserts = [L + c + R for L, R in splits for c in letters]
    # 参考inserts，deletes为将数组拆分为左L和右L，其中右边的元素跳过0位，从下角标第一位开始取，相当于减去了一个元素
    deletes = [L + R[1:] for L, R in splits if R]
    # 参考deletes， replaces为将deletes操作中删掉的元素用letters中的每一个元素来代替。
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]

    # return set(splits)
    return set(inserts + deletes + replaces)


def generate_edit_two(str):
    """
    给定一个字符串,生成编辑距离不大于2的字符串
    """
    return [e2 for e1 in generate_edit_one(str) for e2 in generate_edit_one(e1)]


# 测试
# print(len(generate_edit_two("apple")))
# print(len(generate_edit_two("apple")))
print(len(generate_edit_one("apple")))
print(len(generate_edit_two("apple")))