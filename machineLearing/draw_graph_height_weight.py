import numpy as np
import matplotlib.pyplot as plt

# 第一列身高，第二列体重
data = np.array([[152, 51], [156, 53], [160, 54], [164, 55],
        [168, 57], [172, 60], [176, 62], [180, 65],
        [184, 69], [188, 72]])

# 将体重和升高得值取出存入xy轴中
x, y = data[:, 0].reshape(-1, 1), data[:, 1]

plt.scatter(x, y, color = 'black')
plt.xlabel('height(cm)')
plt.ylabel('weight(kg)')
plt.show()
