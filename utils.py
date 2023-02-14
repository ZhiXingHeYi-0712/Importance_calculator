from typing import Tuple
import numpy as np
import gc

# data shape = (1183, 1679, 1488)

class DataReader():
    def __init__(self) -> None:
        self.data_location = 'data/month_data.npy'

        # 标记内存中存储了哪块数据, 0-7
        self.data_hold_region:int = -1
        self.data: np.ndarray = None      

        # data.shape[1]
        self.Y_ALL: int = 1679

        # 分界点，为了节省内存，对数据进行八分读取。仅在Y轴上分段
        self.Y_GAP: int = self.Y_ALL // 8

        self.y_split_points = [(p+1) * self.Y_GAP for p in range(7)]

    def _getYRegion(self, y: int) -> int:
        '''
        计算y值落在哪个区间
        '''
        region = 0
        for p in self.y_split_points:
            if y >= p:
                region += 1
        return region

    def _getYRange(self, y: int) -> Tuple[int, int]:
        '''
        获取各个区间的Y值范围。
        :param y: y轴索引
        :return: Tuple(起始Index, 结束Index)
        '''
        region = self._getYRegion(y)

        if region == 0:
            return (0, self.Y_GAP)
        elif region == 7:
            return (self.Y_GAP * 7, self.Y_ALL + 1)
        else:
            return (self.Y_GAP * region, self.Y_GAP * (region + 1))

    def _setData(self, y: int) -> None:
        '''
        申请一次数据。数据存储在self.data
        '''
        range_left, range_right = self._getYRange(y)
        region = self._getYRegion(y)

        # 加载数据
        if region == self.data_hold_region:
            # 已经存储
            pass
        else:
            self.data = None
            gc.collect()
            # 未存储
            data_full = np.load(self.data_location)
            self.data = data_full[:, range_left:range_right].copy()

            # 释放内存
            del data_full
            gc.collect()

            self.data_hold_region = region

    def getPixel(self, x, y) -> np.ndarray:
        self._setData(y)
        return self.getPixelFromCurrentData(x, y, self.data)

    def getPixelFromCurrentData(self, x, y, data):
        region = self._getYRegion(y)
        
        # y值的底数，需要减去
        y_base = self.Y_GAP * region 

        y_direct = y - y_base

        return data[x, y_direct]


def isNoData(d: np.ndarray) -> bool:
    return d[0] == -9999 or d[1] == -9999 or d[2] == -9999 or d[-1] == 0 or d[0] == 0 or np.count_nonzero(d) < 800


# 裁剪数组
r = DataReader()
r._setData(0)
np.save('data_clip/data_0.npy', r.data)