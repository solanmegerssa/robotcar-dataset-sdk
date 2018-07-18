################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import argparse
import os
import re
import matplotlib.pyplot as plt
import cv2
import numpy as np
from lane_finding import find_lanes, plot_lanes
from lane_to_pose import find_pose, build_image_transform
from datetime import datetime as dt
from image import load_image
from camera_model import CameraModel
from transform import build_se3_transform

parser = argparse.ArgumentParser(description='Play back images from a given directory')
# parse input arguments
parser.add_argument('--dir', type=str, help='Directory containing images.')
parser.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
parser.add_argument('--scale', type=float, default=1.0, help='(optional) factor by which to scale images before display')
parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
args = parser.parse_args()

camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0)
model = None
# get timestamps
timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps'))
if not os.path.isfile(timestamps_path):
  timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps')
  if not os.path.isfile(timestamps_path):
      raise IOError("Could not find timestamps file")

current_chunk = 0
timestamps_file = open(timestamps_path)

# get camera model
if args.models_dir:
    model = CameraModel(args.models_dir, args.dir)

    # get camera extrinsic matrix
    extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    # create vehicle-camera transform
    G = build_se3_transform(extrinsics)
    G_camera_vehicle = np.zeros((3,4))
    G_camera_vehicle[:,:] = G[0:3,:]


    G_image_car = build_image_transform(G_camera_vehicle, model)

for line in timestamps_file:
    tokens = line.split()
    datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
    chunk = int(tokens[1])

    filename = os.path.join(args.dir, tokens[0] + '.png')
    if not os.path.isfile(filename):
        if chunk != current_chunk:
            print("Chunk " + str(chunk) + " not found")
            current_chunk = chunk
        continue

    current_chunk = chunk

    """
    MY OWN CODE FROM HERE ON OUT

    """

    img = load_image(filename, None)

    # check if image is in color
    if len(img[0,0]) > 1:
        color = True
    else:
        color = False
        
    left_lane, right_lane = find_lanes(img, color)
    plot_lanes(img, left_lane, right_lane, datetime)


    """
    future work

    x0 = 0
    test = find_pose(x0,left_lane,right_lane, G_image_car)
    print test
    if test is not None:
        cv2.circle(img,test,2,(240,2,2))
    """

    

