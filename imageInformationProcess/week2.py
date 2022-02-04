import numpy as np
import matplotlib.pyplot as plt
import cv2

low_contrast = 30
plt.show()
n, bins, patches = plt.hist(low_contrast.ravel(), range(256), facecolors='blue', alpha=0.5)
plt.show()
cumulative_bins = n
for i in range(1, len(cumulative_bins)):
    cumulative_bins[i] = cumulative_bins[i] + cumulative_bins[n - 1]
plt.plot(range(len[n]), cumulative_bins)
plt.show()