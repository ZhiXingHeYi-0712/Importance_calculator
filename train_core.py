# 用于进行拟合训练的文件

from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from NDVI_mean import getNDVI_mean_by_lucc
import estimate

# 读取co2数据。由于节省空间，co2数据没有包含在.npy文件中，需要另外读取。
co2_all = np.load('data/WLG/co2.npy')

# 滞后时长。3即为选择包括当前月在内的3个月进行计算，即当前月，前1月，前2月
delay = 3

# NODATA像元值
NODATA = (np.zeros((17, )) - 9999)


def isNoData(d: np.ndarray) -> bool:
    """判断原始数组是不是nodata

    Parameters
    ----------
    d : np.ndarray
        单个像元上的数组，应该是长度为1488的一维数组

    Returns
    -------
    bool
        是否是nodata
    """
    return (d[0] == -9999 or d[1] == -9999 or d[2] == -9999 or d[-1] == 0 or d[0] == 0)



def preprocess(dataset: np.ndarray) -> Tuple[np.ndarray]:
    """进行进一步预处理

    Parameters
    ----------
    dataset : np.ndarray
        单个像元上的数组，应该是长度为1488的一维数组

    Returns
    -------
    Tuple[np.ndarray]
        _description_
    """
    # 如果像元是nodata则直接返回-1
    if isNoData(dataset):
        # nodata pixel
        return (-1,)

    # 像元数据中先分离NDVI和LULC    
    NDVI = dataset[:240]
    LULC = dataset[240:480]

    # 计算各个LULC对应的基底NDVI
    NDVI_mean_base = np.array(list(map(getNDVI_mean_by_lucc, LULC)))

    # 计算NDVI距平
    NDVI_to_mean = NDVI - NDVI_mean_base

    # 再分离出长度均为252的一系列气象数据
    pr_all = dataset[480:732]
    srad_all = dataset[732:984]
    tmmn_all = dataset[984:1236]
    tmmx_all = dataset[1236:]

    # 下面给出的是各气象要素的切片，均为(240, delay)形状的二维数组。请仔细注意切片方法。
    pr   = np.array([pr_all   [i+1:i+delay + 1] for i in range(len(NDVI))])
    srad = np.array([srad_all [i+1:i+delay + 1] for i in range(len(NDVI))])
    tmmn = np.array([tmmn_all [i+1:i+delay + 1] for i in range(len(NDVI))])
    tmmx = np.array([tmmx_all [i+1:i+delay + 1] for i in range(len(NDVI))])
    co2  = np.array([co2_all  [i+1:i+delay + 1] for i in range(len(NDVI))])

    # 组合成features
    features = np.hstack((LULC.reshape(240, 1), co2, pr, srad, tmmn, tmmx))

    # labels就是NDVI距平
    labels = NDVI_to_mean

    # 生成训练数据集
    xTrain, xTest, yTrain, yTest, real_NDVI_train, real_NDVI_test = train_test_split(
        features, labels, NDVI, test_size=0.01, shuffle=True)

    # 这个结果将被传入train()函数
    return (xTrain, xTest, yTrain, yTest, real_NDVI_train, real_NDVI_test, features, labels)


# RF核心函数，还需要兼顾分辨NODATA像元
def train(dataset_after_preprocess):
    """核心训练函数

    Parameters
    ----------
    dataset_after_preprocess : np.ndarray
        被preprocess()函数处理后的ndarray

    Returns
    -------
    np.ndarray
        结果数组
    """
    if len(dataset_after_preprocess) == 1:
        # nodata pixel
        return NODATA

    xTrain, xTest, yTrain, yTest, real_NDVI_train, real_NDVI_test, features, labels = dataset_after_preprocess

    xTrain_no_LULC = xTrain[:, 1:]
    xTest_no_LULC = xTest[:, 1:]

    # 构造RFR回归器，回归次数宜适中，否则训练时长难以接受。评价标准采用误差绝对值
    rfr = RandomForestRegressor(
        n_estimators=50, random_state=114514, oob_score=True, criterion='absolute_error')

    # 训练
    rfr.fit(xTrain_no_LULC, yTrain)

    # 调用estimate工具计算重要性
    importance, base_r2 = estimate.getImportance(rfr, xTrain, real_NDVI_train)

    # 保存重要性
    importance = list(importance)
    # 也保存R2备用
    importance.extend([base_r2])

    # 标准长度63，结果扩大100000倍以便后续处理
    return (np.array(importance) * 100000)


def train_main(dataset, x, y):
    '''
    dataset是原始数组
    '''
    # 先执行预处理
    pre = preprocess(dataset)

    # 在结果中记录xy坐标
    result = [x, y]

    # 追加训练结果
    result.extend(train(pre))

    # 打印到控制台
    print(
        f'finish training ({x}, {y}), LULC_imp = {result[2]} CO2 = {sum(result[3:3+delay])} climate = {sum(result[3+delay:-1])} R2 = {result[-1]}')
    
    return result
