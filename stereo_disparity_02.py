### Stereo Disparity 02

##### Import statements

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from helper import *
from skimage.metrics import structural_similarity

##### Loading images and true disparity maps

path = "2021 mobile datasets/artroom1"

calib = read_calib( f"{ path}/calib.txt")
disp_L, scale = read_pfm( f"{ path}/disp0.pfm")
disp_R, _ = read_pfm( f"{ path}/disp1.pfm")
img_L_orig = Image.open( f"{ path}/im0.png")
img_R_orig = Image.open( f"{ path}/im1.png")
img_L = np.array( img_L_orig.convert( "L"))
img_R = np.array( img_R_orig.convert( "L"))

print_calib( calib)

display( img_L_orig, img_R_orig, disp_L, disp_R, scale, calib[ "vmin"], calib[ "vmax"])

##### Compute disparity using block matching

stereo = cv2.StereoBM_create( numDisparities=176, blockSize=7)
disparity = stereo.compute( img_L, img_R)

##### Scale the range of disparity into [0-1]

disparity = np.float32( disparity - disparity.min()) / np.float32( disparity.max())

##### Showing disparity

plt.figure( figsize=( 10, 5))
plt.imshow( disparity, cmap="jet")
plt.axis( "off")
plt.show()

##### Compute SSIM

ssim_L = structural_similarity( disparity, disp_L, data_range=1)
ssim_R = structural_similarity( disparity, disp_R, data_range=1)

print( f"SSIM with Left Disparity: { ssim_L}")
print( f"SSIM with Right Disparity: { ssim_R}")
print( f"Mean SSIM: { np.mean( [ ssim_L, ssim_R])}")