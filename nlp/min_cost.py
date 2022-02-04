from math import inf

# initiate matrix
matrix = [[inf, 10, 10, inf, inf, inf, inf ,inf],
          [inf, inf, inf, 20, inf, inf, inf, inf],
          [inf, inf, inf, inf, 10, inf, inf, inf],
          [inf, inf, inf, inf, inf, 50, inf, inf],
          [inf, inf, inf, inf, inf, 20, inf, inf],
          [inf, inf, inf, inf, inf, inf, 10, inf],
          [inf, inf, inf, inf, inf, inf, inf, 10],
          [inf, inf, inf, inf, inf, inf, inf, inf]
          ]
# node count
n = len(matrix)
# initiate dp graph
table = [[inf] * n for x in range(n)]
# 设置起点start，
table[0][0] = 0

# iterate from 1
for i in range(1, n):
    # compute every sub cost from v to n
    for v in range(n):
        table[i][v] = table[i - 1][v]  # 起始状态
        for k in range(n):
            # current cost = min (pre cost)
            table[i][v] = min(table[i][v], matrix[k][v] + table[i - 1][k])

print(table[-1])