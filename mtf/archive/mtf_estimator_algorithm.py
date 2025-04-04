# -*- coding: utf-8 -*-

# taken from https://github.com/JorgeGIlG/MTF_Estimator

"""
Copyright 2020 Jorge Gil 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Robust ESF, PSF, FWHM & MTF estimation from low quality targets and synthetic edge creation. 
"""
try:
    from osgeo import gdal
except ImportError:
    import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, interpolate, ndimage, stats
from scipy.optimize import OptimizeWarning


class Edge:
    Cols = None
    Rows = None
    Angle = None
    EdgeFileName = None
    Fwhm = None
    SuperSampFactor = 1000
    Dark = np.iinfo(np.uint16).max * 0.2
    Bright = np.iinfo(np.uint16).max * 0.8

    def __init__(self, edgeFileName, fWhm, angle=10, cols=500, rows=100):
        self.Cols = cols
        self.Rows = rows
        self.Fwhm = fWhm
        self.Angle = np.float64(angle - 90)
        self.EdgeFileName = edgeFileName
        self.create()

    def gaussian(self, x, a, b, c, w):
        fLog2 = -4 * np.log(2)
        return a + b * np.power(np.e, fLog2 * np.power(x - c, 2) / np.power(w, 2))

    def create(self):
        angle = np.float64(self.Angle) * np.pi / 180
        scols = self.Cols * self.SuperSampFactor
        fwhm = self.Fwhm  # px
        sFwhm = fwhm * self.SuperSampFactor
        superGaussian = self.gaussian(
            np.linspace(
                -5 * self.SuperSampFactor,
                5 * self.SuperSampFactor,
                10 * self.SuperSampFactor,
                dtype=np.float64,
            ),
            0,
            1,
            0,
            sFwhm,
        )
        superGaussian = superGaussian / np.sum(superGaussian)

        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(
            self.EdgeFileName, self.Cols, self.Rows, 1, gdal.GDT_UInt16
        )
        band = dst_ds.GetRasterBand(1)

        for row in range(0, self.Rows):

            superEdgePos = np.int(
                np.round(
                    scols // 2
                    + 0.5 * self.SuperSampFactor * (2 * row - self.Rows) / np.tan(angle)
                )
            )

            superEdge = (
                np.ones([scols + 2 * superGaussian.shape[0]], dtype=np.float64)
                * self.Bright
            )
            superEdge[superEdgePos:] = self.Dark
            edge = np.zeros(
                self.Cols
                + np.int(np.ceil(2 * superGaussian.shape[0] / self.SuperSampFactor)),
                dtype=np.float64,
            )

            for col in range(0, self.Cols + 2):
                edge[col] = np.sum(
                    superEdge[
                        col * self.SuperSampFactor : col * self.SuperSampFactor
                        + superGaussian.shape[0]
                    ]
                    * superGaussian
                )

            edge = edge[0 : self.Cols]
            edge = np.round(edge)

            band.WriteArray(np.expand_dims(edge, axis=0), 0, row)

        band = None
        dst_ds = None


def sigmoid(x, a, b, l, s):
    return a + b * (1 / (1 + np.power(np.e, -l * (x + s))))


class Transect:
    __X = None
    __Y = None
    __IsValid = True
    __Snr = None
    __SigmoidParams = None
    # __MinPxs = np.float64(10) # Minimum acceptable PSF half-width (Not FWHM)
    __MinPxs = np.float64(5)  # Minimum acceptable PSF half-width (Not FWHM)
    Row = None
    EdgePx = None
    EdgeSubPx = None
    Plot = True

    def __init__(self, x, y, row, logfunc=None):
        # Clean nodata values
        self.__X = np.float64(x[y != 0])
        self.__Y = np.float64(y[y != 0])
        self.Row = np.float64(row)

        if not logfunc is None:
            self.console = logfunc

        if type(self.__X) != np.ndarray or self.__X.shape[0] <= 2 * self.__MinPxs:
            self.console("Not enough pixels in the transect. Set to invalid.")
            self.__IsValid = False
            return None

        self.__getEdgePx()

    def console(self, *message):  # May be overriden
        message = [str(i) for i in message]
        print(" ".join(message))

    def __getEdgePx(self):
        ySmooth = ndimage.filters.gaussian_filter(self.__Y, 1)

        grad = np.abs(np.diff(ySmooth) / np.diff(self.__X))
        maxPx = self.__X[:-1][grad == np.max(grad)]

        # If there are more than one
        maxPx = [np.round(np.average(maxPx))]

        if (
            maxPx - np.min(self.__X) < self.__MinPxs
            or np.max(self.__X) - maxPx < self.__MinPxs
        ):
            self.console("Not enough pixels to build a PSF. Set to invalid")
            self.__IsValid = False
            return None

        self.EdgePx = maxPx[0]

    """
    def getSnr(self):
        if not self.__IsValid:
            return None
        
        edgeIdx = np.where(self.__X == self.EdgePx)[0][0]
                
        l = self.__Y[:edgeIdx] 
        r = self.__Y[edgeIdx:]
        
        lAvg = np.average(l)
        rAvg = np.average(r)
        
        if lAvg < rAvg:
            self.__Increase = True
        else:
            self.__Increase = False
        
        self.__Snr = np.abs(lAvg-rAvg)/np.sqrt(np.power(np.std(l),2)+np.power(np.std(r),2))
        return self.__Snr
    """

    def sigmoidFit(self, initGuess):
        if initGuess is None:
            initGuess = [np.min(self.__Y), np.max(self.__Y), 1.0, -self.EdgePx]

        try:
            popt, pcov = optimize.curve_fit(sigmoid, self.__X, self.__Y, p0=initGuess)
            self.__SigmoidParams = popt
            self.EdgeSubPx = -self.__SigmoidParams[3]
        except OptimizeWarning:
            return False, False
        except:
            return False, False

        """        
        if self.Plot:
            a, b, l, s = popt
            #plt.figure()
            x = np.arange(np.min(self.__X), np.max(self.__X), step=1e-2)
            plt.plot(self.__X, self.__Y, ".")
            plt.plot(x, sigmoid(x, a, b, l, s), "-")
            #print s
            #plt.show()
        """

        return popt, pcov

    def getRefinedData(self):
        if not self.__IsValid:
            plt.close()
            raise Exception("Invalid transects")

        a, b, l, s = self.__SigmoidParams
        return np.array([self.__X - self.EdgeSubPx, (self.__Y - a) / b])

    def isValid(self):
        return self.__IsValid

    def invalidate(self):
        self.__IsValid = False

    def getInitGuess(self):
        return self.__SigmoidParams


class Mtf:

    __PreRefinementEdgeSubPx = None
    __RefineEdgeSubPxStep = 0
    Image = None
    ResultsStr = ""
    # Transects = list()
    LVarThresh2 = np.float64(3e-2)  # Squared variance threshold for l
    OverSampFreq = np.float64(1e3)  # Samples per pixel
    PsfMaxHalfWidth = np.float64(10)  # Pixels
    Figure = None
    SubPlot = None
    Plot = True

    def __init__(self, imagePath, logfunc=None):

        if not logfunc is None:
            self.console = logfunc

        self.Transects = list()

        image = self.readImage(imagePath)
        self.Image = image
        rows, cols = image.shape

        # Prepare plot
        if self.Plot:
            self.Figure, self.SubPlot = plt.subplots(2, 2)
            self.Figure.subplots_adjust(hspace=0.2, wspace=0.2)

        # Create an initial list of valid transects
        initGuess = None
        x = np.float64(np.arange(0, cols))
        for i in range(0, rows):
            r = image[i, :]
            self.console("Row:", i)
            t = Transect(x, r, i, logfunc=logfunc)

            # Find subpx edge position
            if t.isValid():
                popt, pcov = t.sigmoidFit(initGuess)

                if popt is False:
                    t.invalidate()
                    self.console("Unable to fit row")
                    continue

                if pcov[2][2] < self.LVarThresh2:
                    initGuess = t.getInitGuess()
                    self.Transects.append(t)
                else:
                    t.invalidate()
                    self.console("Set to invalid due to bad 'l' covariance")

        self.console("Found ", len(self.Transects), "valid transects out of ", rows)

        for i in range(
            0, 2
        ):  # First: Remove outliers. Second: Recalculate linear regression.
            self.refineEdgeSubPx()

        lsfData = self.getEsfData()
        lsf = self.calcOptimizedLsf(lsfData)
        self.calcMtf(lsf)

    def console(self, *message):  # May be overriden
        message = [str(i) for i in message]
        print(" ".join(message))

    # Refine by linear regression
    def refineEdgeSubPx(self):
        x = None
        y = None

        for t in self.Transects:
            if x is None:
                x = np.array([t.Row])
                y = np.array([t.EdgeSubPx])
            else:
                x = np.append(x, t.Row)
                y = np.append(y, t.EdgeSubPx)

        b, a, r, p, stderr = stats.linregress(x, y)

        self.console("Refined subpx edge pos. Coefficient of correlation: ", r**2)

        diff = y - (a + b * x)
        # avg = np.average(diff)
        std = np.std(diff)

        self.console("STEP: ", self.__RefineEdgeSubPxStep)

        if self.__RefineEdgeSubPxStep == 0:  # Remove outliers
            transects = list()
            for t in self.Transects:
                if np.abs(a + b * t.Row - t.EdgeSubPx) > 1.75 * std:
                    self.console("Removed outlier", t.Row)
                    t.invalidate()
                else:
                    transects.append(t)
            self.__PreRefinementEdgeSubPx = np.array([y, x], dtype=np.float64)
            self.__RefineEdgeSubPxStep = 1
            self.Transects = transects
            self.console("Remaining transects: ", len(self.Transects))
        else:  # Set new subpixel edge pos
            self.__RefineEdgeSubPxStep = 2
            for t in self.Transects:
                t.EdgeSubPx = a + b * t.Row

        if len(self.Transects) < 5:
            plt.close()
            raise Exception("Not enough transects")

        if self.__RefineEdgeSubPxStep == 2:

            self.ResultsStr += "Angle: %f\n" % -(np.arctan(b) * 180 / np.pi)

            if self.Plot:
                self.SubPlot[0, 0].imshow(self.Image)
                # self.SubPlot[0,0].plot(self.__PreRefinementEdgeSubPx[0],self.__PreRefinementEdgeSubPx[1], "+", color="black")
                self.SubPlot[0, 0].plot(y, x, "+", color="black")
                xAux = np.arange(np.min(x), np.max(x), step=1e-3)
                self.SubPlot[0, 0].plot(a + b * xAux, xAux, "-", color="black")
                self.SubPlot[0, 0].set_title("Edge image")
                self.SubPlot[0, 0].axes.set_xlim(left=0, right=self.Image.shape[1])
                self.SubPlot[0, 0].axes.set_ylim([0, self.Image.shape[0]])

    def getEsfData(self):
        esfData = None
        for t in self.Transects:
            if esfData is None:
                esfData = t.getRefinedData()
            else:
                esfData = np.append(esfData, t.getRefinedData(), axis=1)

        esfData = np.sort(esfData, axis=1)
        filter = np.logical_and(
            [esfData[0] >= -self.PsfMaxHalfWidth], [esfData[0] <= self.PsfMaxHalfWidth]
        )[0]
        return np.compress(filter, esfData, axis=1)

    """
    Find best spline smoothing factor and gaussian
    by using a Levenberg-Marquardt optimization
    """

    def calcOptimizedLsf(self, esfData):

        self.console("Optimizing LSF")

        fLog2 = -4 * np.log(2)

        def gaussianFunc(x, a, b, c, w):
            return a + b * np.power(np.e, fLog2 * np.power(x - c, 2) / np.power(w, 2))

        def costFunc(params):
            smooth, ga, gb, gc, gw = params
            lsfRep = interpolate.splrep(x, y, k=3, s=smooth)
            psfSpline = interpolate.splev(xAux, lsfRep, der=1)
            return np.sum(np.abs(psfSpline - gaussianFunc(xAux, ga, gb, gc, gw)))

        def fwhm_from_lsf(x, y):  # Instead of of the Gaussian model
            c = np.where(y == np.max(y))[0][0]  # x value for maximum, center
            y = np.abs(y - np.max(y) / 2.0)
            left = np.where(y[:c] == np.min(y[:c]))
            right = np.where(y[c:] == np.min(y[c:])) + c
            left = x[left[0][0]]
            right = x[right[0][0]]
            return right - left

        x = esfData[0]
        y = esfData[1]

        xAux = np.arange(np.min(x), np.max(x), step=1 / self.OverSampFreq)

        initGuess = [np.min(y), np.max(y), 1.0, 0]
        popt, pcov = optimize.curve_fit(sigmoid, x, y, p0=initGuess)
        a, b, l, s = popt

        # m = np.float64(esfData.shape[1])

        # x0 = [(m - np.sqrt(2*m))*1e-5, a, b/2, s, 2]
        x0 = [1e-9, a, b / 2, s, 2]
        # bounds = [(1e-12,None),(0,1),(0,1),(-self.PsfMaxHalfWidth,self.PsfMaxHalfWidth),(-self.PsfMaxHalfWidth,self.PsfMaxHalfWidth)]
        bounds = [
            (1e-10, 0.2),
            (0, 0.1),
            (0, 3),
            (-self.PsfMaxHalfWidth, self.PsfMaxHalfWidth),
            (-self.PsfMaxHalfWidth, self.PsfMaxHalfWidth),
        ]
        opt = optimize.minimize(
            costFunc,
            x0,
            args=(),
            method="L-BFGS-B",
            jac=None,
            bounds=bounds,
            tol=None,
            callback=None,
            options={
                "disp": None,
                "maxls": 20,
                "iprint": -1,
                "gtol": 1e-05,
                "eps": 1e-08,
                "maxiter": 15000,
                "ftol": 2.220446049250313e-09,
                "maxcor": 10,
                "maxfun": 15000,
            },
        )
        optSmooth, ga, gb, gc, gw = opt["x"]

        self.ResultsStr += "Smooth: %e \n" % optSmooth

        lsfRep = interpolate.splrep(x, y, k=3, s=optSmooth)
        lsfSpline = interpolate.splev(xAux, lsfRep, der=1)

        """
        # Write the LSF to a file for external analysis        
        oFile = open("/tmp/lsf.csv", "w")
        oFile.write("pixel, lsf\n")
        for i in np.array([xAux,lsfSpline]).T:
            oFile.write("%.24e,%.24e\n" % (i[0], i[1]))
        oFile.close()
        """

        self.ResultsStr += "FWHM: %f px\n" % gw  # From estimated Gaussian

        if self.Plot:
            esfSpline = interpolate.splev(xAux, lsfRep)
            self.SubPlot[0, 1].plot(esfData[0], esfData[1], "+")
            self.SubPlot[0, 1].plot(xAux, sigmoid(xAux, a, b, l, s), "-", color="black")
            self.SubPlot[0, 1].plot(xAux, esfSpline, "-", color="red")

            lsfPlot = self.SubPlot[0, 1].twinx()
            # self.SubPlot[0,1].plot(xAux, lsfSpline,"-", color="blue")
            # self.SubPlot[0,1].plot(xAux, gaussianFunc(xAux, ga, gb, gc, gw),"-", color="brown")
            lsfPlot.plot(xAux, lsfSpline, "-", color="blue")
            lsfPlot.plot(xAux, gaussianFunc(xAux, ga, gb, gc, gw), "-", color="brown")
            self.SubPlot[0, 1].set_title("ESF & LSF estimation")

        return np.array([xAux, lsfSpline])

    def calcMtf(self, lsf):

        # If needed, remove the last element of the PSF to get an even number of elements
        if lsf.shape[1] / 2.0 != lsf.shape[1] // 2.0:
            lsf = lsf[:, :-1]

        sampFreq = np.float64(lsf.shape[1]) / (np.max(lsf[0]) - np.min(lsf[0]))

        lsf = lsf[1]
        n = lsf.shape[0]

        lsf = np.append(np.append(np.zeros([20 * n]), lsf), np.zeros([20 * n]))

        lsf = lsf / np.sum(lsf)
        n = np.float64(lsf.shape[0])
        mtf = np.fft.rfft(lsf)
        mtfFreq = np.linspace(0, 0.5 * sampFreq, num=mtf.shape[0], dtype=np.float64)

        mtfVsFreq = interpolate.interp1d(mtfFreq, np.absolute(mtf), kind="linear")
        freqVsMtf = interpolate.interp1d(np.absolute(mtf), mtfFreq, kind="linear")
        self.ResultsStr += "MTF0: %s \n" % mtf[0]
        self.ResultsStr += "MTF30: %f \n" % freqVsMtf(0.3)
        self.ResultsStr += "MTF50: %f \n" % freqVsMtf(0.5)
        self.ResultsStr += "MTF@Nyquist: %f \n" % mtfVsFreq(0.5)
        self.console(
            "\n############ Results\n" + self.ResultsStr + "####################\n"
        )

        if self.Plot:
            mtf = mtf[mtfFreq <= 0.6]
            mtfFreq = mtfFreq[mtfFreq <= 0.6]
            self.SubPlot[1, 0].plot(mtfFreq, np.absolute(mtf), "-")
            self.SubPlot[1, 0].set_title("MTF modulus estimation")
            self.SubPlot[1, 1].axis("off")
            self.SubPlot[1, 1].text(0.1, 0.1, str(self.ResultsStr))
            plt.show(block=False)

    def readImage(self, imagePath):
        if isinstance(imagePath, str):
            ds = gdal.Open(imagePath, gdal.GA_ReadOnly)
        else:
            ds = imagePath
        rows = ds.RasterYSize
        cols = ds.RasterXSize
        band = ds.GetRasterBand(1)
        image = np.float64(band.ReadAsArray(0, 0, cols, rows))
        band = None
        ds = None
        return image


ImageFile = "2022-12-15 edge in water f6.tif"

"""
angle="16.776550"
fwhm="2.101313"
imgFile = "/ram/myedge_angle_"+str(angle)+"_fwhm_"+str(fwhm)+".tif"
edge = Edge(imgFile, np.float64(fwhm), cols=500, rows=100, angle=np.float64(angle))
ImageFile = imgFile
"""

mtf = Mtf(ImageFile)


# exit(0)
