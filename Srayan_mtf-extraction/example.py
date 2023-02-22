# adapted from
# https://github.com/u-onder/mtf.py
# Copyright (c) 2021 Othneil Drew
# under MIT licence

import mtf as mtf
from skimage import io
import matplotlib.pyplot as plt

data_dir = "../../data_store/"
bg_dir = data_dir + "/2023-01-17 background images"
img_dir = data_dir + "/2023-01-17 edge redone"

img_path = img_dir + "/2023-01-17 edge in water f-a"

x_min = 500
x_max = 1000
y_min = 300
y_max = 700

imgArr = io.imread(img_path + "/MMStack_Pos0.ome.tif")[63, x_min:x_max, y_min:y_max]
# imgArr = mtf.Helper.LoadImageAsArray(img_path+'/MMStack_Pos0.ome.tif')
plt.imshow(imgArr)
plt.show()
res = mtf.MTF.CalculateMtf(imgArr, verbose=mtf.Verbosity.DETAIL)
