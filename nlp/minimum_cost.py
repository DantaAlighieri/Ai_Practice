# from nlp.stack import Stack


def minimum_cost(matrix, start, end, n):
    start_to_current_cost = [8]
    start_to_pre_cost = [8]

    node_size = len(matrix)

    start_to_current_cost[0] = 0
    start_to_pre_cost[0] = 0

    # 需要一个队列
    stack = []
    stack.append(start)
    seen = set()  # 看是否访问过
    seen.add(start)
    while (len(stack) > 0):
        # 拿出邻接点
        vertex = stack.pop()  # 这里pop参数没有0了，最后一个元素
        nodes = matrix[vertex]
        for w in nodes:
            if w not in seen:  # 如何判断是否访问过，使用一个数组
                stack.append(w)
                seen.add(w)
        print(vertex)

    # stack = Stack()
    # stack.push(0)
    # for i in range(len(matrix)):
    #     row = stack.pop()
    #
    #     for j in range(len(matrix)):
    #         if i != j and matrix[i][j] > 0 :
    #
    #         if(matrix[i][j] != -1)
    #
    #
    #
    # while(stack.size() != 0):
    #     node = stack.pop()
    #     column = matrix[:node]
    #     for i in range(len(column)):
    #         if(column[i] != -1 && visted[i] != -1):
    #             stack.push(column[i])


# 定义一个图的结构
matrix = {
    "start": ["A", "B"],
    "A": ["C"],
    "B": ["D"],
    "C": ["E"],
    "D": ["E"],
    "E": ["F"],
    "F": ["End"],
    "End" : []
}

minimum_cost(matrix, "start", "end", 8)
