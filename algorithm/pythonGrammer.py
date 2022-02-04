import sys
import numpy as np

# # practice
# a = [0 for _ in range(20)]
# print(a)
# b = [[0 for _ in range(3)] for _ in range(4)]
# print(b)
# # practice
#
# arr = [1, 2, 3]
# m = len(arr)
# n = 6
#
# for j in range(m):
#     print(arr[j])
#
# C = 5
# table = [0 for i in range(C + 1)]
# print(table)

# x = 5
# y = 3
# dtw = np.zeros((x, y))
# print(dtw)


# 定义距离
def euc_dist(v1, v2):
    return np.abs(v1 - v2)


s = [3, 4, 5]
t = [5, 6, 7]
m = len(s)
n = len(t)

dtw = np.zeros((m, n))
dtw.fill(sys.maxsize)

dtw[0, 0] = euc_dist(s[0], t[0])
for ii in range(1, m):
    dtw[ii, 0] = dtw[ii - 1, 0] + euc_dist(s[ii], t[0])
print(dtw)
for ii in range(1, n):
    dtw[0, ii] = dtw[0, ii - 1] + euc_dist(s[0], t[ii])
print(dtw)


test = np.ones((3, 2))
print(test)