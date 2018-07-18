Robotcar Dataset Python Tools
=============================

This is a forked repo from the Oxford Self Driving Data set. My goal was to build a function to detect lanes using image recognition. 

I used their function which reads images as a starting place.

Requirements
------------
The python tools have been tested on Python 2.7. 
Python 3.* compatibility has not been verified.

The following packages are required:
* numpy
* matplotlib
* pillow
* colour_demosaicing

These can be installed with pip:

```
pip install numpy matplotlib colour_demosaicing pillow
```

Data
------------------
I tested on the Oxford small sample data set:

Download by clicking here, then extract all files in the same directory:
http://robotcar-dataset.robots.ox.ac.uk/downloads/sample_small.tar

Command Line Tools
------------------

### Detect Lanes
The `detect_lanes.py` script can be used to view images and plots lanes from the data-set.

```bash
python detect_lanes.py --images_dir /path/to/data/stereo/centre
```


Design Choices
------------------
I implemented a lane detection function which takes input images and then plots the images with the lanes overlaid.

The function is broken down into 4 main parts:

1: Edge detection
- I used the Canny edge detecting algorithm to find edges in the original image. Without this, the next steps would be much harder.

2: Masking
- In this case, I'm only really interested in the portion of the image which contains the lanes. Thus, I mask out everything above the horizon and below the nose of the car so that they don't interfere with the lane detection.

3: Line detection
- I used the probabilistic Hough Line Transform to extact lines among the edges. The proabilistic implementation is more efficient, and is also nice because it outputs actual pixel locations for each line.

4: Lane detection
- Finally, I wanted to group together the detected lines into a right lane and a left lane so that I could plot the lanes more clearly. First I rejected lines with slopes near horizontal, then divided the lines that were on the right and left side of the FOV. I performed linear regression on the points combining the left and right lanes to arrive on my final estimate for the lanes. This is what is then plotted


Results
------------------
My function had trouble detecting the left lane, although this is because there wasn't really a distinguishable lane in most of the images. Detection of the right lane was better because the lane was more apparent.