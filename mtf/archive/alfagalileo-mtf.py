from __future__ import division
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from matplotlib.patches import Rectangle
import numpy as np
from PIL import *


class IMaGE(object):
    def __init__(self, fit=False):
        self.ax = plt.gca()
        self.rect = Rectangle(
            (0, 0), 1, 1, antialiased=True, color="b", linestyle="solid", lw=1.2
        )
        self.rect.set_fill(False)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.fit = fit

        self.key = False
        self.count = 0

        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.ax.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        self.count += 1
        self.key = True if self.key % 2 == 0 else False

        if self.key % 2 == 0:
            self.x = range(int(self.x0), int(self.x1))
            self.cut(im, x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)
            self.ESF()
            self.LSF()
            self.MTF()

        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_motion(self, event):
        if self.key:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            self.ax.figure.canvas.draw()

    def cut(self, image, **args):
        for i in args.keys():
            print("{} : {}".format(i, args[i]))

        M = image[int(args["y0"]) : int(args["y1"]), int(args["x0"]) : int(args["x1"])]
        self.M_out = 0.299 * M[:, :, 0] + 0.589 * M[:, :, 1] + 0.114 * M[:, :, 2]

        # plt.figure()
        # plt.title("operator box-area")
        # plt.imshow(self.M_out)
        # name = "box_selection_{}.png".format(self.count)
        # imag = Image.fromarray(np.asarray(self.M_out),mode = "RGB")
        # imag.save("prueba.png")
        # plt.show()

    def ESF(self):
        """
        Edge Spread Function calculation
        """

        self.X = self.M_out[100, :]
        mu = np.sum(self.X) / self.X.shape[0]
        tmp = (self.X[:] - mu) ** 2
        sigma = np.sqrt(np.sum(tmp) / self.X.shape[0])
        self.edge_function = (self.X[:] - mu) / sigma

        self.edge_function = self.edge_function[::3]
        x = range(0, self.edge_function.shape[0])

        plt.figure()
        plt.title(r"ESF")
        plt.plot(x, self.edge_function, "-ob")
        plt.show()

    def LSF(self):
        """
        Line Spread Function calculation
        """
        self.lsf = self.edge_function[:-2] - self.edge_function[2:]
        x = range(0, self.lsf.shape[0])

        # plt.figure()
        # plt.title("LSF")
        # plt.xlabel(r'pixel') ; plt.ylabel('intensidad')
        # plt.plot(x,self.lsf,'-or')
        # plt.show()

    def MTF(self):
        """
        Modulation Transfer Function calculation
        """
        self.mtf = abs(np.fft.fft(self.lsf))
        self.mtf = self.mtf[:] / np.max(self.mtf)
        self.mtf = self.mtf[: len(self.mtf) // 2]
        ix = np.arange(self.mtf.shape[0]) / (self.mtf.shape[0])
        mtf_poly = np.polyfit(ix, self.mtf, 6)
        poly = np.poly1d(mtf_poly)

        plt.figure()
        plt.title("MTF")
        plt.xlabel(r"Frecuency $[cycles/pixel]$")
        plt.ylabel("mtf")
        (p,) = plt.plot(ix, self.mtf, "-or")
        (ll,) = plt.plot(ix, poly(ix))
        plt.legend([p, ll], ["MTF values", "polynomial fit"])
        plt.grid()
        plt.show()


if __name__ == "__main__":
    from skimage import io

    # data_dir = "../../data_store/"
    # bg_dir = data_dir + "/2023-01-17 background images"
    # img_dir = data_dir + "/2023-01-17 edge redone"

    # img_path = img_dir + "/2023-01-17 edge in water f-a"

    plt.figure()
    plt.title("Testing Image")
    plt.xlabel(r"M")
    plt.ylabel(r"N")
    im = io.imread("2023-01-17 edge in water f-a-64-rgb.tif")
    a = IMaGE(fit=True)

    plt.imshow(im, cmap="gray")
    plt.show()
