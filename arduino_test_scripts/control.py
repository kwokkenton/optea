# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from time import sleep
import matplotlib.pyplot as plt
from pycromanager import Core

core = Core()

# core.snap_image()
# tagged_image = core.get_tagged_image()

# image_height = tagged_image.tags['Height']
# image_width = tagged_image.tags['Width']

# image = tagged_image.pix.reshape((image_height, image_width))

# plt.imshow(image, cmap='gray')
# plt.show()

n_angles = 10
for i in range(n_angles):
    core.set_relative_position(2000/n_angles)
    core.set_shutter_open(True)
    sleep(2)