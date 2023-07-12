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

##### Compute disparity using block matching function of opencv

stereo = cv2.StereoBM_create( numDisparities=176, blockSize=7)
disparity = stereo.compute( img_L, img_R)

##### Scale the range of disparity into [0-1]

disparity = np.float32( disparity - disparity.min()) / np.float32( disparity.max())

disp_L_scaled = np.float32( disp_L - disp_L.min()) / np.float32( disp_L.max())

disp_R_scaled = np.float32( disp_R - disp_R.min()) / np.float32( disp_R.max())

##### Showing disparity

plt.figure( figsize=( 10, 5))
plt.imshow( disparity, cmap="jet")
plt.axis( "off")
plt.show()

##### Compute SSIM

ssim_L = structural_similarity( disparity, disp_L_scaled, data_range=1)
ssim_R = structural_similarity( disparity, disp_R_scaled, data_range=1)

print( f"SSIM with Left Disparity: { ssim_L}")
print( f"SSIM with Right Disparity: { ssim_R}")
print( f"Mean SSIM: { np.mean( [ ssim_L, ssim_R])}")

##### Implementation of block matching algorithm

##### Filter function

# params:
# 	a: Src
# 	b: Kernel
# 	cost: Matching Cost Function
# returns:
#	x: location corresponding to the minimum value of cost
def filter( a, b, cost):
	_, s_w = a.shape
	_, k_w = b.shape

	R = np.zeros( s_w - k_w + 1)

	for i in range( s_w - k_w + 1):
		R[ i] = cost( a[ :, i: i + k_w], b)

	x = np.argmin( R)
	
	x = int( x + ( k_w / 2))

	return x

##### Matching cost functions

# SAD (Sum of Absolute Difference)

def SAD( a, b):
	c = np.abs( np.array( a) - np.array( b))
	return c.sum()

# SSD (Sum of Squared Difference)

def SSD( a, b):
	c = np.square( np.array( a) - np.array( b))
	return c.sum()

# NCC (Normalized Cross Correlation)

def NCC( a, b):
	c = cv2.matchTemplate( np.float32( a), np.float32( b), cv2.TM_CCORR_NORMED)
	# negate the result because filter function looks for minimum value
	return -np.max( c)

# LoG (Laplacian of Gaussian) and SAD

def LoG_SAD( a, b):
	a = cv2.Laplacian( np.float64( a), cv2.CV_64F)
	b = cv2.Laplacian( np.float64( b), cv2.CV_64F)

	return SAD( a, b)

# LoG (Laplacian of Gaussian) and SSD

def LoG_SSD( a, b):
	a = cv2.Laplacian( np.float64( a), cv2.CV_64F)
	b = cv2.Laplacian( np.float64( b), cv2.CV_64F)

	return SSD( a, b)

##### Block matching algorithm

# params:
# 	left: Left Image
# 	right: Right Image
# 	cost: Matching Cost Function
#	numDisparity: Disparity Range
#	ksize: Size of the Kernel
# returns:
#	D: Disparity Map
def block_matching( left, right, cost, numDisparity, ksize=3):
	height, width = left.shape

	# Radius of Kernel
	r = int( ksize / 2)

	D = np.zeros( ( height, width))

	for i in range( height):
		for j in range( width):
			# Region Of Interest in Left Image
			roi_height_low = i - r if i - r > -1 else 0
			roi_width_low = j - r if j - r > -1 else 0

			# Select Out the Region from Left Image
			roi = left[ roi_height_low: i + r + 1, roi_width_low: j + r + 1]

			# Select Out the Area of Search from Right Image
			area = right[ roi_height_low: i + r + 1, roi_width_low: j + numDisparity + 1]

			# Apply Filter with Desired Cost Function
			s = filter( area, roi, cost)

			# Displacement
			u = s - r

			D[ i][ j] = u
	
	return D

##### Resizing images

# Almost Quarter of Original Size
new_size = ( 427, 240)

ndisp_resized = int( calib[ "ndisp"] / 4)

disp_L_resized = np.array( Image.fromarray( disp_L).resize( new_size))
disp_R_resized = np.array( Image.fromarray( disp_R).resize( new_size))

img_L_resized = np.array( Image.fromarray( img_L).resize( new_size)) / 255
img_R_resized = np.array( Image.fromarray( img_R).resize( new_size)) / 255

fig = plt.figure( figsize=( 12, 7))

fig.add_subplot( 2, 2, 1)
plt.imshow( img_L_resized)
plt.title( "Left View")
plt.axis( "off")

fig.add_subplot( 2, 2, 2)
plt.imshow( img_R_resized)
plt.title( "Right View")
plt.axis( "off")

fig.add_subplot( 2, 2, 3)
plt.imshow( disp_L_resized, cmap="jet")
plt.title( "Left Disparity")
plt.axis( "off")

fig.add_subplot( 2, 2, 4)
plt.imshow( disp_R_resized, cmap="jet")
plt.title( "Right Disparity")
plt.axis( "off")

plt.show()

##### Scale the range of true resized disparity into [0-1]

disp_L_resized_scaled = np.float32( disp_L_resized - disp_L_resized.min()) / np.float32( disp_L_resized.max())

disp_R_resized_scaled = np.float32( disp_R_resized - disp_R_resized.min()) / np.float32( disp_R_resized.max())

# Disparity using SAD cost function

disp = block_matching( img_R_resized, img_L_resized, SAD, ndisp_resized)

# scale the range of disparity into [0-1]
disp = np.float32( disp - disp.min()) / np.float32( disp.max())

ssim_L = structural_similarity( disp_L_resized_scaled, disp, data_range=1)
ssim_R = structural_similarity( disp_R_resized_scaled, disp, data_range=1)

print( f"SSIM with Left Disparity: { ssim_L}")
print( f"SSIM with Right Disparity: { ssim_R}")
print( f"Mean SSIM: { np.mean( [ ssim_L, ssim_R])}")

plt.figure( figsize=( 10, 5))
plt.imshow( disp, cmap="jet")
plt.title( "Disparity Using SAD Cost Function")
plt.axis( "off")
plt.show()

# Disparity using SSD cost function

disp = block_matching( img_R_resized, img_L_resized, SSD, ndisp_resized)

# scale the range of disparity into [0-1]
disp = np.float32( disp - disp.min()) / np.float32( disp.max())

ssim_L = structural_similarity( disp_L_resized_scaled, disp, data_range=1)
ssim_R = structural_similarity( disp_R_resized_scaled, disp, data_range=1)

print( f"SSIM with Left Disparity: { ssim_L}")
print( f"SSIM with Right Disparity: { ssim_R}")
print( f"Mean SSIM: { np.mean( [ ssim_L, ssim_R])}")

plt.figure( figsize=( 10, 5))
plt.imshow( disp, cmap="jet")
plt.title( "Disparity Using SSD Cost Function")
plt.axis( "off")
plt.show()

# Disparity using NCC cost function

disp = block_matching( img_R_resized, img_L_resized, NCC, ndisp_resized)

# scale the range of disparity into [0-1]
disp = np.float32( disp - disp.min()) / np.float32( disp.max())

ssim_L = structural_similarity( disp_L_resized_scaled, disp, data_range=1)
ssim_R = structural_similarity( disp_R_resized_scaled, disp, data_range=1)

print( f"SSIM with Left Disparity: { ssim_L}")
print( f"SSIM with Right Disparity: { ssim_R}")
print( f"Mean SSIM: { np.mean( [ ssim_L, ssim_R])}")

plt.figure( figsize=( 10, 5))
plt.imshow( disp, cmap="jet")
plt.title( "Disparity Using NCC Cost Function")
plt.axis( "off")
plt.show()

# Disparity using LoG_SAD cost function

disp = block_matching( img_R_resized, img_L_resized, LoG_SAD, ndisp_resized)

# scale the range of disparity into [0-1]
disp = np.float32( disp - disp.min()) / np.float32( disp.max())

ssim_L = structural_similarity( disp_L_resized_scaled, disp, data_range=1)
ssim_R = structural_similarity( disp_R_resized_scaled, disp, data_range=1)

print( f"SSIM with Left Disparity: { ssim_L}")
print( f"SSIM with Right Disparity: { ssim_R}")
print( f"Mean SSIM: { np.mean( [ ssim_L, ssim_R])}")

plt.figure( figsize=( 10, 5))
plt.imshow( disp, cmap="jet")
plt.title( "Disparity Using LoG-SAD Cost Function")
plt.axis( "off")
plt.show()

# Disparity using LoG_SSD cost function

disp = block_matching( img_R_resized, img_L_resized, LoG_SSD, ndisp_resized)

# scale the range of disparity into [0-1]
disp = np.float32( disp - disp.min()) / np.float32( disp.max())

ssim_L = structural_similarity( disp_L_resized_scaled, disp, data_range=1)
ssim_R = structural_similarity( disp_R_resized_scaled, disp, data_range=1)

print( f"SSIM with Left Disparity: { ssim_L}")
print( f"SSIM with Right Disparity: { ssim_R}")
print( f"Mean SSIM: { np.mean( [ ssim_L, ssim_R])}")

plt.figure( figsize=( 10, 5))
plt.imshow( disp, cmap="jet")
plt.title( "Disparity Using LoG_SSD Cost Function")
plt.axis( "off")
plt.show()