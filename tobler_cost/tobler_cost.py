import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import elevation
import richdem as rd

# get image
img_file = 'image.tif'
tenaya_creek = rd.LoadGDAL(img_file)

# elevation
plt.imshow(tenaya_creek, interpolation='none')
plt.colorbar()
plt.show()

# slope
slope = rd.TerrainAttribute(tenaya_creek[0:512, 0:512], attrib='slope_riserun', zscale=0.00001)
rd.rdShow(slope, axes=False, cmap='magma', figsize=(8, 5.5))
plt.show()

# cost function
cost = 0.6*np.exp(3.5*np.abs(slope + 0.05))
plt.imshow(np.minimum(cost, 20), interpolation='none')
plt.colorbar()
plt.show()