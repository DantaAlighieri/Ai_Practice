import sys
import numpy as np

def difference(n, m):
    return abs(n - m)

def dtw(x, y, x_length, y_length):

    # 初始化
    graph = [[0] * x_length for i in range(y_length)]
    graph[x_length - 1][0] = difference(x[0], y[0])
    # 先初始化x轴
    for num in range(1, x_length):
        graph[x_length - 1][num] = graph[x_length - 1][num - 1] + difference(x[num], y[0])

    # 在初始化y轴
    for num in range(y_length - 1, 0, -1):
        graph[num - 1][0] = graph[num][0] + difference(y[y_length - num], x[0])

    # 填充表格， 从左下角开始横向填充，已知填满表格
    for num_x in range(y_length - 2, -1, -1):
        for num_y in range(1, x_length):
            graph[num_x][num_y] = min(graph[num_x][num_y - 1], graph[num_x + 1][num_y - 1], graph[num_x + 1][num_y]) + difference(x[num_y], y[y_length - num_x - 1])

    printMatrix(graph, x_length, y_length)
    return graph

def printMatrix(graph, x_length, y_length):
    for x in range(0, y_length):
        for y in range(0, x_length):
            print(str(graph[x][y]) + ",", end="")
        print("")

def printMinPath(graph, xArr, yArr, x_length, y_length):
    print("(%d,%d)"%(yArr[y_length - 1], xArr[x_length - 1]))
    y = x_length - 1
    x = 0
    while 1:
        # x轴不能越界
        # y轴不能越界
        # 当x，和y到00点时结束，即graph[y_length -1][0]
        if y == 0 and x == y_length - 1:
            break
        if y - 1 >= 0:
            left = graph[x][y - 1]
        if x + 1 < y_length:
            down = graph[x + 1][y]
        if y - 1 >= 0 and x + 1 < y_length:
            left_down = graph[x + 1][y - 1]
        minValue = min(left, down, left_down)
        if minValue == left_down:
            print("(%d,%d)"%(yArr[y_length - (x + 1) - 1], xArr[y - 1]))
            x+=1
            y-=1
        if minValue == left:
            print("(%d,%d)" % (yArr[y_length - (x + 1)], xArr[y - 1]))
            y-=1
        if minValue == down:
            print("(%d,%d)" % (yArr[y_length - (x + 1) - 1], xArr[y]))
            x+=1


# test
x = [1, 6, 2, 3, 0, 9, 4, 1, 6, 3]
y = [1, 3, 4, 9, 8, 2, 1, 5, 7, 3]

#test
# x = [5, 6, 9]
# y = [5, 6, 7]

x_length = len(x)
y_length = len(y)

printMinPath(dtw(x, y, x_length, y_length), x, y, x_length, y_length)

