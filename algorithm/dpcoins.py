import sys

# 动态规划硬币问题求解
# 有arr=[5，10，11]块的硬币，计算出构成j元需要的最少硬币数量
# 设M[j]为构成j元的最少钱数

def findMinCoinNumber(arr, q):
    coinNumber = len(arr)
    # 初始化，设置从1到q都需要最大的硬币个数，第0个为0
    optimalCount = [-1 for i in range(q + 1)]
    optimalCount[0] = 0
    print(optimalCount)
    for qCount in range(1, q + 1):
        for coinCount in range(coinNumber):
            if qCount >= arr[coinCount]:
                remainQ = qCount - arr[coinCount]
                if optimalCount[remainQ] != -1 and (optimalCount[qCount] == -1 or optimalCount[remainQ] + 1 < optimalCount[qCount]):
                    optimalCount[qCount] = optimalCount[remainQ] + 1
    return optimalCount[q]
arr = [2,5,11]
q = 22
print("凑成%d元钱，需要用到的最少硬币数量为：%s" %(q, findMinCoinNumber(arr, q)))