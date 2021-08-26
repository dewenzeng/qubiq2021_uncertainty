import SimpleITK as sitk
import tifffile as tiff
import numpy as np

a = np.zeros([1,10,100,100,3])
a[:,:,:,:,0] = 255
tiff.imsave('new.tiff',a.astype(np.uint8))