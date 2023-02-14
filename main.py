import numpy as np
from train_core import train_main
import joblib
import numpy as np
import time
import gc


# @@@@@@ 进程数修改，你计算机有几个核就写两倍的数字。例如6核就写12. @@@@@
CORE_NUMBER = 1

# 训练数据组号修改，一共有0-7共8组。根据切割数据方案的不同，组数也不同
TRAIN_LIST = [7]

# 记录时间
now = time.localtime()
nowt = time.strftime("%Y-%m-%d-%H_%M_%S", now)  # 这一步就是对时间进行格式化
print(nowt)


for i in TRAIN_LIST:  # i 是region

    # 读取一组数据，并清理内存以免内存不足
    data = np.load(f'data_clip/data_{i}.npy')
    gc.collect()

    # 开启多进程multi-processing（注意不是多线程multi-thread，多线程在python中不能提高计算效率）
    parallel = joblib.Parallel(n_jobs=CORE_NUMBER) 

    X_shape, Y_shape, _ = data.shape

    # 多进程方式调用训练主函数`train_main()`，得到即为结果。请参考`train_core.py`以了解训练主函数
    result_all = parallel(joblib.delayed(train_main)(
        data[x, y, :], x, y + (209 * i)) for y in range(Y_shape) for x in range(X_shape))

    now = time.localtime()
    nowt = time.strftime("%Y-%m-%d-%H:%M:%S", now)  # 这一步就是对时间进行格式化
    print(f'finish region {i} at {nowt}')
    
    # 保存结果
    np.save(f'result/result_{i}.npy', result_all)


