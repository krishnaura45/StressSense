# -*- coding: utf-8 -*-
# ***Converting images from .npy to .jpg format***
import numpy as np

img_array = np.load('/content/image_1.npy')
# One of the easiest ways to view them is using matplotlib's imshow function:

from matplotlib import pyplot as plt

plt.imshow(img_array, cmap='rainbow')
plt.show()

import numpy as np
import os
img_array = np.load('/content/image_2.npy')
# One of the easiest ways to view them is using matplotlib's imshow function:

from matplotlib import pyplot as plt

plt.imshow(img_array, cmap='rainbow')
plt.show()

i=1
outpath = "/content/Img_d"
while i<4:
  img_array = np.load('/content/image_{0}.npy'.format(i))

  plt.imshow(img_array, cmap='nipy_spectral')

  plt.tight_layout()
  plt.savefig(os.path.join(outpath,"img_{0}.jpg".format(i)))
  plt.show()
  i+=1

!zip -r log.zip /content/Img_d/
