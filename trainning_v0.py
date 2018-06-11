
###############################
# Data acquisition for 3D soft object 
# with optical waveguides sensing system
# Jose Barreiros, Alec Cornwell
# Jun, 2018
################################

import sys, cv2, argparse, time, pdb
import numpy as np 


# imports for maximum filter
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float


def yes_or_no(question):
	while "the answer is invalid":
		reply = str(input(question+' (y/n): ')).lower().strip()
		if reply[0] == 'y':
			return True
		if reply[0] == 'n':
			return False

def getCOM(contours, index):
	# function to get center of "mass" of the target
	# takes contours as its input which are the contours of the target

	cnt = contours[index]
	M = cv2.moments(cnt)				# get moments
	cx = int(M['m10']/M['m00'])			# x center of target
	cy = int(M['m01']/M['m00'])			# y center of target
	return (cx, cy)

def preprocess(image):
	# function preprocesses the image by converting to grayscale, does
	# histograms equalization, and normalizes

	# convert to grayscale

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	size = gray.shape



	# Histograms Equalization
	equ = cv2.equalizeHist(gray)
	# res = np.hstack((gray,equ)) #debug

	# normalize
	norm_image = np.zeros(size)
	norm_image = cv2.normalize(equ, norm_image, alpha=0, beta=1, 
							   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

	cv2.imshow("norm and equ", norm_image)
	cv2.waitKey(1)

	
	return (norm_image, gray, size)

def getBrightest(norm_image):
	# function gets brightest points of waveguides in the image

	# maximum filter
	im = img_as_float(norm_image)

	# image_max is the dilation of im with a 20*20 structuring element
	# It is used within peak_local_max function
	#image_max = ndi.maximum_filter(im, size=100, mode='constant')

	# Comparison between image_max and im to find the coords of local maxima
	
	min_dis=50
	im=cv2.copyMakeBorder(im,min_dis,min_dis,min_dis,min_dis,cv2.BORDER_CONSTANT,value=0)
	cv2.imshow("im bordered", im)
	cv2.waitKey(1)	
	#pdb.set_trace()
	coordinates = peak_local_max(im, min_distance=min_dis, threshold_rel=.20)

	#for coord in coordinates:
	#	if 

	#print(coordinates)


	return (coordinates, im, im.shape, min_dis)

def image_process(coordinates, image, size):
	# Creates a mask from the brightest points in the image and gets the COM of 
	# the brightest points for each waveguide

	# make mask from brightest points
	new_im = np.zeros(size)

	'''cv2.imshow("new_im", new_im)
	cv2.waitKey(0)'''

	# create mask from brightest coordinates
	mask=new_im
	mask[(coordinates[:, 0], coordinates[:, 1])] = 1
	mask = np.array(mask * 255, dtype = np.uint8)		# grayscale

	#print(mask.dtype)

	# close holes in the mask
	mask = cv2.dilate(mask, None, iterations = 6)
	mask = cv2.erode(mask, None, iterations = 1)


	cv2.imshow("premask", mask)
	cv2.waitKey(1)

	# find contours of the brightest points
	contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
								cv2.CHAIN_APPROX_SIMPLE)[-2]
	

	#print(len(contours))

	i = 0
	index = 0
	COM=[]
	for c in contours:
		# if the contour is too small, ignore it
		cArea = cv2.contourArea(c)

		if (cArea > int(image.size/20000)):
			#print(i)	

			# compute the bounding box for the contour, draw it on the frame
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			text = str(i)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(image,text, (x, y), font, 1, (255, 255, 255), 2)
			COM.append(getCOM(contours, index))
			#cv2.circle(image, getCOM(contours, index), 50, (255, 0, 0), 2)
			#print(COM)
			i+=1
		index+=1

	return COM

def sortCOM(COM):
	# function sorts COM of waveguides in order, from left to right in 
	# increasing row order

	# find waveguides in top and bottom row
	topR =[]
	bottomR = []
	#x=0 #debugging
	for cent in COM:
		# in row 1
		if cent[1] < 227:
			topR.append(cent)

		# in row 2
		else:
			bottomR.append(cent)

	#print(topR)
	#print(bottomR)

	# sort top and bottom rows from left to right
	sort_topR = sorted(topR, key=lambda k: k[0])

	sort_bottomR = sorted(bottomR, key=lambda k: k[0])

	# create ordered list of COMs in order of first row from left to right then
	# second row from left to right
	in_order = []
	in_order.extend(sort_topR)
	in_order.extend(sort_bottomR)

	return in_order

def get_mask(image):
	# preprocess

	(norm_image, gray, size) = preprocess(image)

	#test_im = norm_image.copy()

#	pdb.set_trace()
	(coordinates,image,size,border_size) = getBrightest(norm_image)          # brightest waveguide pixels
	cv2.imshow("im bordered", image)
	cv2.waitKey(1)	
	print(size)
    # COM of brightest pixels for each waveguide
#	pdb.set_trace()
	COM = image_process(coordinates, image, size)

	cv2.imshow("image1", image)
	cv2.waitKey(1)

	in_order = sortCOM(COM)        # sort waveguide COMs left to right by row

	#print(in_order)

	#print(norm_image.shape)

    # test to show right COM corresponds to right waveguide
	z=0
	for coord in in_order:
		cv2.circle(image, coord, 50, (255, 0, 0), 2)
		text = str(z)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image,text, coord, font, 1, (255, 0, 0), 2)
		z+=1

	cv2.imshow("image2", image)
	cv2.waitKey(1)

	mask = np.zeros(size, np.uint8)
	for cent in in_order:
		cv2.circle(mask, cent, 50, 255, thickness=-1)
	
	mask=mask[border_size:size[0]-border_size,border_size:size[1]-border_size]
	pdb.set_trace()
	mask_com=np.array(in_order)-border_size
	return mask, mask_com

print("##### Sequence for acquiring training data #####\n")
print("##### for surface press location sensing in a 3D soft object #####\n")


print("Opening Camera\n")
cap = cv2.VideoCapture(1)
time.sleep(1)
ret, frame = cap.read()

print("Origin image size: "+ str(frame.shape))
frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
# image size

print("Re-scaled image size: "+ str(frame.shape))

reply=yes_or_no("Do you want to find the mask?")
if reply==True:
	(mask,com_mask)=get_mask(frame)
else:
	(mask,com_mask)=mask_file

cv2.imshow("mask", mask)
cv2.waitKey(1)

num_seg = input("What is the size of the matrix (# of segment)? ")

for i in range(1,int(num_seg)):
	print("\n### Press segment #%i ###"  %i)
    
	#reply=yes_or_no("Do you want to save this data?")
    
	while True:
		print("acquiring data for segment#%i" %i)
		#Xtr[i]=acquire_features()
		reply=yes_or_no("Do you want to save this data?")
		if reply==True:		
			print("saving data for segment#%i" %i)	
			#save_feature_i(Xtr[i])
			break

print("Training data has been acquired sucessfully")

    
