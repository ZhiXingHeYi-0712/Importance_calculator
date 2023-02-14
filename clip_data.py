from utils import DataReader
import numpy as np
# 注意：这种切割文件的写法并不是很好，建议结合自己的数据重写。这里仅供参考。

points = [219, 428, 637, 846, 1055, 1264, 1473]

d = DataReader()
for i, p in enumerate(points):
    d._setData(p)
    region = i + 1
    np.save(f'data_clip/data_{region}.npy', d.data)