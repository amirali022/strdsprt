from cmath import inf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import struct
import re

def read_pfm( filename):
	with Path( filename).open( "rb") as pfm_file:
		line1, line2, line3 = ( pfm_file.readline().decode( "latin-1").strip() for _ in range( 3))

		assert line1 in ( "PF", "Pf")

		channels = 3 if "PF" in line1 else 1
		width, height = ( int( s) for s in line2.split())
		scale_endianess = float( line3)
		bigendian = scale_endianess > 0
		scale = abs( scale_endianess)

		buffer = pfm_file.read()
		samples = width * height * channels
		assert len( buffer) == samples * 4

		fmt = f'{ "<>"[ bigendian]}{ samples}f'
		decoded = struct.unpack( fmt, buffer)
		shape = ( height, width, 3) if channels == 3 else ( height, width)

		img = np.flipud( np.reshape( decoded, shape))
		img[ img == inf] = 0

		return img, scale

def read_calib( filename):
	with Path( filename).open() as calib:
		d = {}
		d[ "cam0"], d[ "cam1"] = ( np.array( [ float( x) for x in re.findall( r"\d+\.?\d*", calib.readline()[ 5:])]).reshape( ( 3, 3)) for _ in range( 2))
		d[ "doffs"], d[ "baseline"], d[ "width"], d[ "height"], d[ "ndisp"], d[ "vmin"], d[ "vmax"] = ( float( calib.readline().split( "=")[ 1]) for _ in range( 7))
		
		return d
		
def print_calib( calib):
	print( "Left Camera Matrix")
	print( calib[ "cam0"])

	print( "Right Camera Matrix")
	print( calib[ "cam1"])

	print( f"Baseline: { calib[ 'baseline']}")
	
	print( f"Image Size (Width, Height): ({ calib[ 'width']}, { calib[ 'height']})")
	
	print( f"Conservative Bound of Disparity Levels: [0, { calib[ 'ndisp'] - 1}]")

	print( f"Tight Bound of Min and Max Disparities: [{ calib[ 'vmin']}, { calib[ 'vmax']}]")

def display( img_L, img_R, disp_L, disp_R, scale, vmin, vmax):
	fig = plt.figure( figsize=( 12, 7))

	fig.add_subplot( 2, 2, 1)
	plt.imshow( img_L)
	plt.title( "Left View")
	plt.axis( "off")

	fig.add_subplot( 2, 2, 2)
	plt.imshow( img_R)
	plt.title( "Right View")
	plt.axis( "off")

	fig.add_subplot( 2, 2, 3)
	plt.imshow( disp_L / scale, cmap="jet", vmin=vmin / scale, vmax=vmax / scale)
	plt.title( "Left Disparity")
	plt.axis( "off")

	fig.add_subplot( 2, 2, 4)
	plt.imshow( disp_R / scale, cmap="jet", vmin=vmin / scale, vmax=vmax / scale)
	plt.title( "Right Disparity")
	plt.axis( "off")

	plt.show()