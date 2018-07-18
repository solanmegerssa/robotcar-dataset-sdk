################################################################################
#
# Author:
#  Solan Megerssa
#
################################################################################

import matplotlib.pyplot as plt
import cv2
from datetime import datetime as dt
from image import load_image
from camera_model import CameraModel
import numpy as np

def mask_roi(edges):
	"""
	Takes input edges and returns masked edges which gets rid of background

	Params
	------
	edges: array_like

	Returns
	-------
	masked_edges: array_like
		masked edges

	"""

	# pixels of interest
	pixels = np.array([[0,800],[0,600],[400,450],[850,450],[1250,600],[1250,800]], dtype=np.int32)
	masked_edges = np.zeros_like(edges)

	# fill in region
	cv2.fillPoly(masked_edges, [pixels], 255)

	# show masked image
	masked_edges = cv2.bitwise_and(edges,masked_edges)

	return masked_edges

def edge_detect(img, color_image, threshold1=50, threshold2=150):
	"""
	Takes input image and returns edges using Canny algorithm

	Params
	------
	img: array_like

	threshold1: int
		pixel gradient lower threshold

	threshold2: int
		pixel gradient upper threshold

	Returns
	-------
	edges: array_like
		edge array
	"""

	# check if image is color and gray-scales if it is
	if color_image:
		# gray scale
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# edge detection
		edges = cv2.Canny(img, threshold1, threshold2)
		return edges

	else:
		edges = cv2.Canny(img, threshold1, threshold2)

	return edges

def line_detect(edges):
	"""
	Takes input image and returns lines using probabilistic Hough Transform

	Params
	------
	edges: array_like
		detected edges
	
	Returns
	-------
	lines : list
		list of lists of line endpoints eg. [[x1,y1,x2,y2], [x3,y3,x4,y4], ...]
	"""

	minLineLength = 20
	maxLineGap = 20

	lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
	assert lines is not None, "No lines found"
	return lines[0]

def lane_detect(lines):
	"""
	Takes input lines, detects and returns lanes

	Params
	------
	lines: list
		list of detected lines
	
	Returns
	-------
	lanes : list
		list of left and right lane params
	"""

	# left and right lane lists
	left_lane = []
	right_lane = []
	for x1,y1,x2,y2 in lines:
		
		# rejects near-horizontal lines
		m = (float(y2)-y1)/(float(x2)-x1)
		if abs(m) > 0.1:

			# checks if right or left lane
			if x1 < 1250/2:
				left_lane.append([x1,y1])
				left_lane.append([x2,y2])
			else:
				right_lane.append([x1,y1])
				right_lane.append([x2,y2])
	
	# checks if left and right lane have been detected
	if len(left_lane) > 0:
		left_lane = cv2.fitLine(np.array(left_lane), cv2.cv.CV_DIST_L2, 0 ,.01, .01)
	else:
		left_lane = None
	if len(right_lane) > 0:
		right_lane = cv2.fitLine(np.array(right_lane), cv2.cv.CV_DIST_L2, 0, .01, .01)
	else:
		right_lane = None
	lanes = [left_lane, right_lane]

	return lanes

def find_lanes(img, color_image = True):
	"""
	Takes input image and returns detected lanes

	Params
	------
	img: array_like
		unprocessed car camera image
	
	Returns
	-------
	lanes : list
		list of lists of lane endpoints eg. [[x1,y1,x2,y2], [x3,y3,x4,y4], ...]
	"""

	# check img exists
	assert type(img) == np.ndarray, "img does not exist"
	
	edges = edge_detect(img, color_image)
	mask_edges = mask_roi(edges)
	lines = line_detect(mask_edges)
	left_lane, right_lane = lane_detect(lines)

	return left_lane, right_lane

def plot_lanes(img, left_lane,right_lane, datetime):
	"""
	Takes input image, lanes, and date, and plots them together

	Params
	------
	img: array_like
		unprocessed car camera image

	left_lane: list
		lane line parameters

	right_lane: list
		lane line parameters

	datetime: string
	
	
	"""

	lane_img = np.zeros_like(img)

	# check if right and left lanes exist
	if left_lane is not None:
		vx,vy,x0,y0 = left_lane
		m = 400
		cv2.line(lane_img,(x0-m*vx[0], y0-m*vy[0]), (x0+m*vx[0], y0+m*vy[0]), (0,255,0), 6)

	if right_lane is not None:
		vx,vy,x0,y0 = right_lane
		m = 400
		cv2.line(lane_img,(x0-m*vx[0], y0-m*vy[0]), (x0+m*vx[0], y0+m*vy[0]), (30,230,30), 6)
	
	# plot lanes on top of image
	a,b,g = 0.8,1.,0.
	img = cv2.addWeighted(img,a,lane_img,b,g)
	plt.imshow(img, cmap='gray')
	plt.xlabel(datetime)
	plt.xticks([])
	plt.yticks([])
	plt.pause(0.01)
	