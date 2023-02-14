# 用于计算贡献率的文件
import numpy as np
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics

from NDVI_mean import getNDVI_mean_by_lucc
delay = 3

def predict(rfr: RandomForestRegressor, X: np.ndarray):
    LULC = X[:, 0]
    X_no_LULC = X[:, 1:]
    
    NDVI_to_mean_predict = rfr.predict(X_no_LULC)
    NDVI_mean_base = np.array(list(map(getNDVI_mean_by_lucc, LULC.astype('int32'))))
    
    NDVI_predict = NDVI_mean_base + NDVI_to_mean_predict
    return NDVI_predict

def getR2Score(rfr: RandomForestRegressor, X: np.ndarray, y: np.ndarray) -> float:
    """计算R2

    Parameters
    ----------
    rfr : RandomForestRegressor
        训练好的RFR
    X : np.ndarray
        features
    y : np.ndarray
        labels

    Returns
    -------
    float
        R2
    """
    NDVI_predict = predict(rfr, X)
    return sklearn.metrics.r2_score(y, NDVI_predict)
    
def getRandomFactor(estimate_factor_index: int, lulc_kind_count, data_length: int = 237):
    """MDA法中要求随机生成某个变量

    Parameters
    ----------
    estimate_factor_index : int
        需要被评估的因子下标
    lulc_kind_count : _type_
        _description_
    data_length : int, optional
        _description_, by default 237

    Returns
    -------
    _type_
        _description_
    """
    if estimate_factor_index < lulc_kind_count:
        # 土地利用类型
        return np.random.randint(1, 18, size=data_length)
    elif estimate_factor_index < lulc_kind_count + delay:
        # CO2
        return np.random.randint(340, 440, size=data_length)
    elif estimate_factor_index < lulc_kind_count + (delay * 2):
        # 降水
        return np.random.randint(0, 500, size=data_length)
    elif estimate_factor_index < lulc_kind_count + (delay * 3):
        # srad
        return np.random.randint(0, 300, size=data_length)
    else:
        # 气温
        return np.random.randint(50, 400, size=data_length)

def getOneFactorImportance(rfr: RandomForestRegressor, X: np.ndarray, y: np.ndarray, estimate_factor_index: int):
    # 先随机一个被模拟因子
    random_factor = getRandomFactor(estimate_factor_index, 1)
    
    # 更换
    true_value = np.copy(X[:, estimate_factor_index])
    X[:, estimate_factor_index] = random_factor

    # 再计算R2
    result = getR2Score(rfr, X, y)

    # 要记得换回来，不然下次就是两个量被更换了
    X[:, estimate_factor_index] = true_value
    return result


def getImportance(rfr: RandomForestRegressor, X, y):
    # 先转为int类型
    y = y.astype('int32')

    # 计算不更改因子时的基础R2
    base_r2 = getR2Score(rfr, X, y)
    
    factor_count = X.shape[1]
    result = []

    for factor in range(factor_count):
        # 评估单个变量的贡献率，factor是因子的序号（下标）
        result.append(getOneFactorImportance(rfr, X, y, factor))

    # 归一化处理
    delta = np.abs(np.array(result) - base_r2)
    return ((delta / delta.sum()), base_r2)
