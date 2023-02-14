import numpy as np

result = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0
}

result = [0] * 18
pixel_count = [0] * 18

for ds in range(8):
    data = np.load(f'data_clip/data_{ds}.npy')
    
    x_shape, y_shape, _ = data.shape
    
    for x in range(x_shape):
        for y in range(y_shape):
            LULC = data[x, y, 240:480]
            if np.any(LULC > 0):
                NDVI = data[x, y, :240]
                for i in range(240):
                    result[LULC[i]] += (NDVI[i] / 10000)
                    pixel_count[LULC[i]] += 1
    print(f'finish dataset {ds}')
print(result)
print(pixel_count)

