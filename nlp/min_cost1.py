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
cost_graph = [[inf] * n for x in range(n)]
cost_graph[0][0] = 0

# iterate from 1
for i in range(1, n):
    # compute every sub cost from v to n
    for v in range(n):
        cost_graph[i][v] = cost_graph[i - 1][v]
        for k in range(n):
            # current cost = min (pre cost)
            cost_graph[i][v] = min(cost_graph[i][v], matrix[k][v] + cost_graph[i - 1][k])

print(cost_graph[-1][n - 1])