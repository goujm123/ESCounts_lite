import os
import numpy as np

# full = np.load("stu1_10_full.npz")['arr_0']
# d0 = np.load("stu1_10_0.npz")['arr_0']
# d1 = np.load("stu1_10_1.npz")['arr_0']
# d2 = np.load("stu1_10_2.npz")['arr_0']
# d3 = np.load("stu1_10_3.npz")['arr_0']
#
# data = np.concatenate([d0, d1, d2, d3])
# diff = np.abs(full - data)

demo = np.load("train951_demo.npz")['arr_0']
data = np.load("train951.npz")['arr_0']

diff = np.abs(demo - data)
diff_max = np.max(diff)
diff_avg = np.average(diff)

print("diff max = ", diff_max)

# 实际测试结果，分段加载（chunk_size=16） 和 一次性加载(以stu1_10.mp4为例， chunk_size设为60)数据是一致的
