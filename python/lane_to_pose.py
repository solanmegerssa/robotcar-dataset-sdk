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


def build_image_transform(G_camera_car, camera_model):
	"""
	Creates image-car transform

	Params
	------

	G_camera_car: 4x4 array
		Camera-car transformation

	camera_model: Camera
		Camera() class instance

	Returns
	-------
	pose: 3x4 array
		image-car transform
	"""

	fx,fy = camera_model.focal_length
	cx,cy = camera_model.principal_point

	A = np.array([[fx, fx, cx], [0, fy, cy],[0, 0, 1]])
	T = G_camera_car

	K = np.dot(A,T)

	return K

def pixel2car(pixel, G_image_car):
	"""
	Converts pixel u,v coordinates into coords relative to car

	Params
	------
	pixel: 2x1 array

	G_camera_car: 4x4 array
		Camera-car transformation

	Returns
	-------
	pose: 3x1 array
		pose relative to car
	"""
	pixel_hom = np.ones(3,1)
	pixel_hom[0:1,1] = pixel
	xyz = np.linalg.solve(np.inv(G_image_car), pixel)
	print xyz
	return xyz


def find_pose(x0, left_lane, right_lane, G_image_car):
	"""
	Takes previous pose, left and right lanes, and returns the interpolated pose

	Params
	------
	x0: 3x1 array

	left_lane: list
		params for left lane

	right_lane: list
		params for right lane

	G_camera_car: 4x4 array
		Image-car transformation

	Returns
	-------
	pose: 3x1 array
		pose relative to x0
	"""
	if left_lane is not None and right_lane is not None:
		vx,vy,x0,y0 = left_lane
		vx,vy,x1,y1 = right_lane
		return (x1-x0)/2, (y1-y0)/2

	else:
		return None
