import matplotlib
import tensorflow as tf
import numpy as np

MAX = 1  # Max value of a pixel
GAUS_KERNEL = 3
GAUS_SIGMA = 1.2


def Gaussian_Filter(kernel_size=GAUS_KERNEL, sigma=GAUS_SIGMA):  # Default: Filter_shape = [5,5]
	# 	--> Reference: https://en.wikipedia.org/wiki/Canny_edge_detector#Gaussian_filter
	k = (kernel_size - 1) // 2
	filter = []
	sigma_2 = sigma ** 2
	for i in range(kernel_size):
		filter_row = []
		for j in range(kernel_size):
			Hij = np.exp(-((i + 1 - (k + 1)) ** 2 + (j + 1 - (k + 1)) ** 2) / (2 * sigma_2)) / (2 * np.pi * sigma_2)
			filter_row.append(Hij)
		filter.append(filter_row)

	return np.asarray(filter).reshape(kernel_size, kernel_size, 1, 1)


"""
 NOTE: 	All variables are initialized first for reducing proccessing time.
"""
gaussian_filter = tf.constant(Gaussian_Filter(), tf.float32)  # STEP-1
h_filter = tf.reshape(tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32), [3, 3, 1, 1])  # STEP-2
v_filter = tf.reshape(tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], tf.float32), [3, 3, 1, 1])  # STEP-2

np_filter_0 = np.zeros((3, 3, 1, 2))
np_filter_0[1, 0, 0, 0], np_filter_0[1, 2, 0, 1] = 1, 1  ### Left & Right
# print(np_filter_0)
filter_0 = tf.constant(np_filter_0, tf.float32)
np_filter_90 = np.zeros((3, 3, 1, 2))
np_filter_90[0, 1, 0, 0], np_filter_90[2, 1, 0, 1] = 1, 1  ### Top & Bottom
filter_90 = tf.constant(np_filter_90, tf.float32)
np_filter_45 = np.zeros((3, 3, 1, 2))
np_filter_45[0, 2, 0, 0], np_filter_45[2, 0, 0, 1] = 1, 1  ### Top-Right & Bottom-Left
filter_45 = tf.constant(np_filter_45, tf.float32)
np_filter_135 = np.zeros((3, 3, 1, 2))
np_filter_135[0, 0, 0, 0], np_filter_135[2, 2, 0, 1] = 1, 1  ### Top-Left & Bottom-Right
filter_135 = tf.constant(np_filter_135, tf.float32)

np_filter_sure = np.ones([3, 3, 1, 1]);
np_filter_sure[1, 1, 0, 0] = 0
filter_sure = tf.constant(np_filter_sure, tf.float32)
border_paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])


def Border_Padding(x, pad_width):
	for _ in range(pad_width): x = tf.pad(x, border_paddings, 'SYMMETRIC')
	return x


def FourAngles(d):
	d0 = tf.compat.v1.to_float(tf.greater_equal(d, 157.5)) + tf.compat.v1.to_float(tf.less(d, 22.5))
	d45 = tf.compat.v1.to_float(tf.greater_equal(d, 22.5)) * tf.compat.v1.to_float(tf.less(d, 67.5))
	d90 = tf.compat.v1.to_float(tf.greater_equal(d, 67.5)) * tf.compat.v1.to_float(tf.less(d, 112.5))
	d135 = tf.compat.v1.to_float(tf.greater_equal(d, 112.5)) * tf.compat.v1.to_float(tf.less(d, 157.5))
	# return {'d0':d0, 'd45':d45, 'd90':d90, 'd135':d135}
	return (d0, d45, d90, d135)


"""
	NOTES: 
	- Input ('img_tensor'): shape = [batch_size, height, width, 1] (This version supports only 1 channel)
	- Output: a batch of images with pixels are either 1 (edge) or 0 (non-edge)
"""


def TF_Canny(img_tensor, minRate=0.10, maxRate=0.20,
			 preserve_size=True, remove_high_val=False, return_raw_edges=False):
	""" STEP-0 (Preprocessing):
		1. Scale the tensor values to the expected range ([0,1])
		2. If 'preserve_size': As TensorFlow will pad by 0s for padding='SAME',
							   it is better to pad by the same values of the borders.
							   (This is to avoid considering the borders as edges)
	"""
	img_tensor = (img_tensor / tf.reduce_max(img_tensor)) * MAX
	if preserve_size: img_tensor = Border_Padding(img_tensor, (GAUS_KERNEL - 1) // 2)

	""" STEP-1: Noise reduction with Gaussian filter """
	x_gaussian = tf.nn.convolution(img_tensor, gaussian_filter, padding='VALID')
	### Below is a heuristic to remove the intensity gradient inside a cloud ###
	if remove_high_val: x_gaussian = tf.clip_by_value(x_gaussian, 0, MAX / 2)

	""" STEP-2: Calculation of Horizontal and Vertical derivatives  with Sobel operator 
		--> Reference: https://en.wikipedia.org/wiki/Sobel_operator	
	"""
	if preserve_size: x_gaussian = Border_Padding(x_gaussian, 1)
	Gx = tf.nn.convolution(x_gaussian, h_filter, padding='VALID')
	Gy = tf.nn.convolution(x_gaussian, v_filter, padding='VALID')
	G = tf.sqrt(tf.square(Gx) + tf.square(Gy))
	BIG_PHI = tf.atan2(Gy, Gx)
	BIG_PHI = (BIG_PHI * 180 / np.pi) % 180  ### Convert from Radian to Degree
	D_0, D_45, D_90, D_135 = FourAngles(BIG_PHI)  ### Round the directions to 0, 45, 90, 135 (only take the masks)

	""" STEP-3: NON-Maximum Suppression
		--> Reference: https://stackoverflow.com/questions/46553662/conditional-value-on-tensor-relative-to-element-neighbors
	"""

	""" 3.1-Selecting Edge-Pixels on the Horizontal direction """
	targetPixels_0 = tf.nn.convolution(G, filter_0, padding='SAME')
	isGreater_0 = tf.compat.v1.to_float(tf.greater(G * D_0, targetPixels_0))
	isMax_0 = isGreater_0[:, :, :, 0:1] * isGreater_0[:, :, :, 1:2]
	### Note: Need to keep 4 dimensions (index [:,:,:,0] is 3 dimensions) ###

	""" 3.2-Selecting Edge-Pixels on the Vertical direction """
	targetPixels_90 = tf.nn.convolution(G, filter_90, padding='SAME')
	isGreater_90 = tf.compat.v1.to_float(tf.greater(G * D_90, targetPixels_90))
	isMax_90 = isGreater_90[:, :, :, 0:1] * isGreater_90[:, :, :, 1:2]

	""" 3.3-Selecting Edge-Pixels on the Diag-45 direction """
	targetPixels_45 = tf.nn.convolution(G, filter_45, padding='SAME')
	isGreater_45 = tf.compat.v1.to_float(tf.greater(G * D_45, targetPixels_45))
	isMax_45 = isGreater_45[:, :, :, 0:1] * isGreater_45[:, :, :, 1:2]

	""" 3.4-Selecting Edge-Pixels on the Diag-135 direction """
	targetPixels_135 = tf.nn.convolution(G, filter_135, padding='SAME')
	isGreater_135 = tf.compat.v1.to_float(tf.greater(G * D_135, targetPixels_135))
	isMax_135 = isGreater_135[:, :, :, 0:1] * isGreater_135[:, :, :, 1:2]

	""" 3.5-Merging Edges on Horizontal-Vertical and Diagonal directions """
	edges_raw = G * (isMax_0 + isMax_90 + isMax_45 + isMax_135)
	edges_raw = tf.clip_by_value(edges_raw, 0, MAX)

	### If only the raw edges are needed ###
	if return_raw_edges: return tf.squeeze(edges_raw)

	""" STEP-4: Hysteresis Thresholding """
	edges_sure = tf.compat.v1.to_float(tf.greater_equal(edges_raw, maxRate))
	edges_weak = tf.compat.v1.to_float(tf.less(edges_raw, maxRate)) * tf.compat.v1.to_float(tf.greater_equal(edges_raw, minRate))

	edges_connected = tf.nn.convolution(edges_sure, filter_sure, padding='SAME') * edges_weak
	for _ in range(10): edges_connected = tf.nn.convolution(edges_connected, filter_sure, padding='SAME') * edges_weak

	edges_final = edges_sure + tf.clip_by_value(edges_connected, 0, MAX)
	return tf.squeeze(edges_final)

def gaussFilter(fx, fy, sigma):
	x = tf.range(-int(fx / 2), int(fx / 2) + 1, 1)
	y = x
	Y, X = tf.meshgrid(x, y)

	sigma = -2 * (sigma ** 2)
	z = tf.cast(tf.add(tf.square(X), tf.square(Y)), tf.float32)
	k = 2 * tf.exp(tf.divide(z, sigma))
	k = tf.divide(k, tf.reduce_sum(k))
	return k


def gaussian_blur(image, filtersize, sigma):
	if len(image.shape) == 3 and image.shape[2] != 3:
		raise TypeError('Incorrect number of channels.')

	elif len(image.shape) == 3 and image.shape[2] == 3:
		n_channels = 3

	elif len(image.shape) == 2:
		image = tf.expand_dims(image, 2)
		n_channels = 1

	fx, fy = filtersize[0], filtersize[1]
	fil = gaussFilter(fx, fy, sigma)
	fil = tf.stack([fil] * n_channels, axis=2)
	fil = tf.expand_dims(fil, 3)

	new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
	new = tf.expand_dims(new, 0)
	res = tf.nn.depthwise_conv2d(new, fil, strides=[1, 1, 1, 1], padding="SAME")

	res = tf.squeeze(res, 0)

	if n_channels == 1:
		res = tf.squeeze(res, 2)

	res = tf.image.convert_image_dtype(res, tf.dtypes.uint8)
	return res


def get_gradients(image, sigma):
	image = gaussian_blur(image, [5, 5], sigma)
	image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
	image = tf.expand_dims(image, 0)
	image = tf.expand_dims(image, 3)
	sobel = tf.image.sobel_edges(image)
	sobel = tf.squeeze(sobel, 0)
	gx = sobel[:, :, :, 0]
	gx = tf.squeeze(gx, 2)

	gy = sobel[:, :, :, 1]
	gy = tf.squeeze(gy, 2)

	Mag = tf.sqrt(tf.add(tf.square(gx), tf.square(gy)))
	Grad = tf.atan2(gy, gx)
	return Mag, Grad


def nonmaxsupress(Gm, Gd):
	nms = np.zeros(Gm.shape)
	h, w = Gm.shape
	Gm = Gm.numpy()
	Gd = Gd.numpy()
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			angle = np.rad2deg(Gd[i, j]) % 180
			mag = Gm[i, j]

			if (0 <= angle < 22.5) or (157.5 <= angle < 180):
				dx, dy = 0, 1
			elif (22.5 <= angle < 67.5):
				dx, dy = 1, -1
			elif (67.5 <= angle < 112.5):
				dx, dy = 1, 0
			elif (112.5 <= angle < 157.5):
				dx, dy = 1, 1

			if mag > Gm[i - dx, j - dy] and mag > Gm[i + dx, j + dy]: nms[i, j] = mag
	return nms

def threshold2(NMS, l, hi):
	h, w = NMS.shape
	mNMS = np.max(NMS)
	T_Low = l * mNMS
	T_High = hi * mNMS
	res1 = np.copy(NMS)
	res = np.asarray(res1,dtype=np.int32)
	for i in range(1, h - 1):
		for j in range(1, w - 1):

			if (res1[i,j]<T_Low):
				res[i,j] = 0
			elif (res1[i,j]>T_High):
				res[i,j] = 128
			elif (res1[i+1,j]>T_High) or (res1[i-1,j]>T_High) or (res1[i,j+1]>T_High) or (res1[i,j-1]>T_High) or (res1[i-1,j-1]>T_High) or (res1[i-1,j+1]>T_High) or (res1[i+1,j+1]>T_High) or (res1[i+1,j-1]>T_High):
				res[i, j] = 255
			#elif (res1[i,j]<T_low):
			#	res[i,j] = 0


	#res[res1 < T_Low] = 0
	#res[res1 > T_High] = 255
	res = tf.convert_to_tensor(res, dtype=tf.uint8)
	return res


def canny_edge(image, sigma, low, high):
	if len(image.shape) == 3 and image.shape[2] == 3:
		raise TypeError('Please input Grayscaled Image.')
	elif len(image.shape) > 3:
		raise TypeError('Incorrect number of channels.')

	Gm, Gd = get_gradients(image, sigma)
	nms = nonmaxsupress(Gm, Gd)
	ret = threshold2(nms, low, high)
	return ret
if __name__ == '__main__': # Test the above code
	import cv2
	#import tkinter
	import matplotlib.pyplot as plt

	from tensorflow.python.ops import io_ops
	matplotlib.use('TkAgg')
	image = io_ops.read_file("./Lenabmp-used-for-testing-purposes-Resolution-320x240-pixels-24-bit-RGB-Size-230454_Q320.jpg")
	color2 = tf.io.decode_png(image, channels=3, dtype=tf.dtypes.uint8, name=None)
	img = tf.image.convert_image_dtype(color2, tf.dtypes.float32)
	img = tf.image.rgb_to_grayscale(img)
	img = tf.image.convert_image_dtype(img, tf.dtypes.uint8)
	x = tf.image.convert_image_dtype(img, tf.dtypes.float32)/255.0
	gray2 = tf.squeeze(img, 2)
	gray3 = tf.squeeze(x, 2)
	x_tensor = tf.expand_dims(tf.expand_dims(gray3, axis=0), -1)
	canny2= TF_Canny(x_tensor)


	canny = canny_edge(gray2, 1.5, 0.05, 0.1)
	plt.imshow(canny2, cmap="gray")


	plt.show()
