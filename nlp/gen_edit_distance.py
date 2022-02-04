def gen_edit_distance_1(word):
    """
    给定一个字符串,生成编辑距离为1的字符串列表。
    个人觉得处理方式有点函数式编程的思维的感觉，类似于haskell的书写方法
    """
    # 用于替换的字母序列
    letters = 'abcdefghijklmnopqrstuvwxyz'
    # 假设单词为apple
    # word[:1]从0取到下角标第1位 = a， word[:2]从0取到下角标第2位= ap
    # word[1:]从下角标第1位开始取, 取到单词末尾 = pple, word[2:]从下角标第2位开始取，取到末尾 = pple
    # 创建二维数组 splits [('', 'apple'), ('a', 'pple'), ('ap', 'ple'), ('app', 'le'), ('appl', 'e'), ('apple', '')]
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    # inserts 代码解释：
    # 将splits数组分为左L和右R两部分，将letters数组进行切割，每个的单词元素为c，inserts数组为，splits数组左 加 letters每一个字母 加 splits数组右构成的数组
    # 例： splits第一个元素：('', 'apple') --> [''(左) + a + 'apple'(右), ''(左) + b + 'apple'(右), ... ''(左) + z + 'apple'(右)]
    # 例： splits第二个元素：('a', 'pple') --> ['a'(左) + a + 'pple'(右), 'a'(左) + b + 'pple'(右), ... 'a'(左) + z + 'pple'(右)]
    # ...
    # 例： splits最后个元素：('apple', '') --> ['apple'(左) + a + ''(右), 'apple'(左) + b + ''(右), ... 'apple'(左) + z + ''(右)]
    inserts = [L + c + R for L, R in splits for c in letters]
    # 参考inserts，deletes为将数组拆分为左L和右L，其中右边的元素跳过0位，从下角标第一位开始取，相当于减去了一个元素 if R => R不能为空，不然会报错
    deletes = [L + R[1:] for L, R in splits if R]
    # 参考deletes， replaces为将deletes操作中删掉的元素用letters中的每一个元素来代替。
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]

    # 将上面三种情况生成的所有可能的词汇返回
    return set(inserts + deletes + replaces)


def gen_edit_distance_2(word):
    # 给定一个字符串,生成编辑距离为2的字符串列表。：相当于遍历在编辑距离为1的基础上生成的字符串列表，对每一个单词再执行一次编辑距离为1的计算，从而得到的词汇列表 下面的程序可以分解为， 第一步先遍历for
    # word_distance_1 in gen_edit_distance_1(str)，获取到编辑距离为1的词组列表word_distance_1，然后遍历结果将其再带入到 for word_distance_2 in
    # gen_edit_distance_2(word_distance_1)编辑距离为1的函数里面，对每一个上一步的结果进行再一次编辑距离计算。最后得到的词组就是word_distance_2 编辑距离为2的词组集合
    return [word_distance_2 for word_distance_1 in gen_edit_distance_1(word) for word_distance_2 in gen_edit_distance_1(word_distance_1)]

# test 打印编辑距离为1的词组和编辑距离为2的词组，比较比较长度
print(len(gen_edit_distance_1("apple")))
print(len(gen_edit_distance_2("apple")))