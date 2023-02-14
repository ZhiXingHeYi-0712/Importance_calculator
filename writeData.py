from osgeo import gdal
import numpy as np

results = [np.load(f'data_clip/data_{i}.npy') for i in range(0, 8)]

result_all = np.hstack(results)

rows = result_all.shape[0]
columns = result_all.shape[1]
dim = result_all.shape[2]

# np.save('result.npy', result_all)

driver = gdal.GetDriverByName('GTiff')

dst_ds = driver.Create('./data_clip_all.tif', columns, rows, dim, gdal.GDT_Float32)

for i in range(dim):
    dst_ds.GetRasterBand(i+1).WriteArray(result_all[:, :, i])

dst_ds.FlushCache()
