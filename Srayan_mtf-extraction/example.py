# adapted from
# https://github.com/u-onder/mtf.py
# Copyright (c) 2021 Othneil Drew
# under MIT licence

import mtf as mtf

imgArr = mtf.Helper.LoadImageAsArray("2022-12-15 edge in water f6.tif")
res = mtf.MTF.CalculateMtf(imgArr, verbose=mtf.Verbosity.DETAIL)
