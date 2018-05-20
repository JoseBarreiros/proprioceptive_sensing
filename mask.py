#####################################
#Waveguide Imaging pre-processing
#Image to signals
#@Jose Barreiros, 2018
#####################################

# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import numpy as np
import pdb as pdb
import matplotlib.pyplot as plt
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=10,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
ap.add_argument("-s", "--source", type=int, default=1,
	help="CAmera source")

args = vars(ap.parse_args())
source=args["source"]

# created a *threaded *video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=source).start()
fps = FPS().start()
i=0

n_frames=args["num_frames"]
while fps._numFrames < n_frames:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	if i==0:
		frame = vs.read()
		frame = imutils.resize(frame, width=810)
		frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		size=frame.shape
		frames=np.zeros(size+(n_frames,))	

	
	frame = vs.read()
	frame = imutils.resize(frame, width=810)
	#pdb.set_trace()
	frames[:,:,i]=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

	# update the FPS counter
	fps.update()
	i=i+1

frame_avg=frames.mean(axis=(2))
#plt.imshow(frame_avg)
#plt.show()
cv2.imwrite('init.png', frame_avg)
cv2.imshow("image", frame_avg)
cv2.waitKey(1)

#pdb.set_trace()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


