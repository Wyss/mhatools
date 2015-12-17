import numpy as np
import scipy
from scipy import ndimage
import tifffile as tf
from skimage import data, io, filters
from sklearn.mixture import GMM

the_file = 'Z:/Synthetic Biology/arcgt/Dave_ArcGTpictures/150924_BFSK_T2S_celegans_embryos/fc0/scan/00-N2-0/CY3_0000.tiff'
the_file_out = 'Z:/Synthetic Biology/arcgt/Dave_ArcGTpictures/150924_BFSK_T2S_celegans_embryos/fc0/scan/00-N2-0/Asobel_CY3_0000.tiff'

img = tf.imread(the_file)

classif = GMM(n_components=2, covariance_type='full')
classif.fit(img.reshape((img.size, 1)))
threshold = np.mean(classif.means_)
binary_img = img > threshold
img = threshold*binary_img
# im = 255*binary_img.astype('int8')
# tf.imsave(the_file_out, img)

im = img.astype('int32')
dx = ndimage.sobel(im, 0)  # horizontal derivative
dy = ndimage.sobel(im, 1)  # vertical derivative
mag = np.hypot(dx, dy)  # magnitude
mag *= 255 / np.max(mag)  # normalize (Q&D)
mag = mag.astype('int8')
tf.imsave(the_file_out, mag)