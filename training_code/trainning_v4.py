###############################
# Data acquisition for 3D soft object 
# with optical waveguides sensing system
# Jose Barreiros, Alec Cornwell
# Jun, 2018
################################

import sys, cv2, argparse, time, pdb, pickle
import numpy as np 


# imports for maximum filter
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float

class Feature(object):
	def __init__(self,frame, gray, pix_cent_gray, pix_cent_bgr, mean_intensity, pixels_bgr, pixels_gray,segmentID):
		self.frame=frame
		self.gray=gray
		self.pix_cent_gray=pix_cent_gray
		self.pix_cent_bgr=pix_cent_bgr
		self.mean_intensity=mean_intensity
		self.pixels_bgr=pixels_bgr
		self.pixels_gray=pixels_gray	
		self.segmentID=segmentID
#		self.coordinate=coordinate			
	
	def __str__(self):
		rep = "Feature Object for segment" + str(self.segmentID) + "\n"
		rep+= "pix_cent_gray:" + str(self.pix_cent_gray) + "\n"
		rep+= "pix_cent_bgr:" + str(self.pix_cent_bgr) + "\n"
		rep+= "mean_intensity:" + str(self.mean_intensity) + "\n"
		rep+= "pixels_bgr:" + str(self.pixels_bgr) + "\n"
		rep+= "pixels_gray:" + str(self.pixels_gray) + "\n"
		return rep

	#frame, gray, pix_cent_gray,pix_cent_bgr, mean_intensity, pixels_bgr,pixels_gray
def yes_or_no(question):
	while "the answer is invalid":
		reply = str( input(question+' (y/n): ')).lower().strip()

		if reply == 'y':
			return True
		elif reply == 'n':
			return False
		else:
			#print("else")
			pass

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
	coordinates = peak_local_max(im, min_distance=min_dis, threshold_rel=.30)

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
	mask = cv2.dilate(mask, None, iterations = 8)
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

def sortCOM(COM, size):
	# function sorts COM of waveguides in order, from left to right in 
	# increasing row order

	#print ("size[0]: ", size[0])

	realY = size[0] - 30
	# find waveguides in top and bottom row
	R1 =[]
	R2 = []
	R3 = []
	R4 =[]
	#x=0 #debugging
	for cent in COM:
		# in row 1
		if cent[1] < (realY // 4):
			R1.append(cent)

		# in row 2
		elif (realY // 4) < cent[1] < (realY // 2):
			R2.append(cent)

		# in row 3
		elif (realY // 2) < cent[1] < ((3 * realY) // 4):
			R3.append(cent)

		# in row 4
		else:
			R4.append(cent)

	'''print R1
	print R2
	print R3
	print R4'''

	# sort rows from left to right
	sort_R1 = sorted(R1, key=lambda k: k[0])

	sort_R2 = sorted(R2, key=lambda k: k[0])

	sort_R3 = sorted(R3, key=lambda k: k[0])

	sort_R4 = sorted(R4, key=lambda k: k[0])

	# create ordered list of COMs in order of R1 from left to right then
	# R2 from left to right and so on
	in_order = []
	in_order.extend(sort_R1)
	in_order.extend(sort_R2)
	in_order.extend(sort_R3)
	in_order.extend(sort_R4)


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

	in_order = sortCOM(COM,size)        # sort waveguide COMs left to right by row

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
	size=mask.shape
	#pdb.set_trace()
	mask_com=np.array(in_order)-border_size
	tuples=tuple(map(tuple, mask_com))
    #mask_com[:]=[x-border_size for x in mask_com]
	mask_com=[]
	for tupleu in tuples:
		mask_com.append(tupleu)
	#print(mask_com)
	return (mask, mask_com, size)

def intensity_cent(in_order, frame, gray):
	# Intensity at center of circle (numpy uses row by column (so x corresponds
	# to column, y corresponds to row))
    

	pix_cent_gray = []
	pix_cent_bgr = []
	for cent in in_order:
		pix_cent_gray.append(gray[cent[1], cent[0]])
		pix_cent_bgr.append(frame[cent[1], cent[0]])
	return (pix_cent_gray,pix_cent_bgr)

def intensity_avg(in_order, gray):
	# Mean intensity of each circle
	mean_intensity = []
	#pdb.set_trace()
	for cent in in_order:
		circle_img = np.zeros((gray.shape), np.uint8)
		cv2.circle(circle_img, cent, 50, (255, 255, 255), thickness=-1)
		#masked_data = cv2.bitwise_and(test_im, test_im, mask=circle_img)

		#cv2.imshow("masked", circle_img)
		#cv2.waitKey(0)

		mean_intensity.append(cv2.mean(gray, mask=circle_img)[::-1])
	return mean_intensity

def getPixels(image, size, in_order):
	intensity_values_from_original = []
	for cent in in_order:
		mask = np.zeros(size, np.uint8)
		cv2.circle(mask, cent, 50, 255, thickness=-1)

		#this will give you the coordinates of points inside the circle
		#cv2.imshow("maskkk", mask)
		#cv2.waitKey(0)
		#pdb.set_trace()
		where = np.where(mask == 255)

		#print where

		intensity_values_from_original.append(image[where[0], where[1]])

	return intensity_values_from_original

def feature_extraction(frame, com_mask,size):
	#image, in_order, gray, test_im, size):
	# find intensities
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	(pix_cent_gray, pix_cent_bgr) = intensity_cent(com_mask, frame,gray)

	mean_intensity = intensity_avg(com_mask, gray)

	pixels_bgr = getPixels(frame, size, com_mask)
	pixels_gray = getPixels(gray, size, com_mask)

	return (frame, gray, np.array(pix_cent_gray),np.array(pix_cent_bgr), np.array(mean_intensity), np.array(pixels_bgr),np.array(pixels_gray))



def main():
	print("##### Sequence for acquiring training data #####\n")
	print("##### for surface press location sensing in a 3D soft object #####\n")


	print("Opening Camera\n")
	cap = cv2.VideoCapture(0)
	#cap.set(3,270) #width
	#cap.set(4,480) #height
	#cap.set(5,1024) #fps

	time.sleep(1)
	ret, frame = cap.read()

	print("Origin image size: "+ str(frame.shape))
	frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
	# image size
	print("Re-scaled image size: "+ str(frame.shape))
	i=0
	reply=yes_or_no("Do you want to find the mask?")
	if reply==True:
        
		while True:
			i=i+1
			ret, frame = cap.read()
			frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
			(mask,com_mask,size)=get_mask(frame)
			cv2.imshow("mask "+str(i), mask)
			print(i)
			cv2.waitKey(1)
			reply1=yes_or_no("Do you want to save this mask?")
			if reply1==True:
				name_f = input("\nName your mask file: ") 
				f=open(name_f+".mask","wb")         
				pickle.dump(mask,f)
				pickle.dump(com_mask,f)
				f.close()
				break
	else:
		name_f = input("What file do you want to open as mask: ")
		f=open(name_f,"rb")
		mask=pickle.load(f)
		com_mask=pickle.load(f)
		size=mask.shape
		#pdb.set_trace()
		cv2.imshow("loaded mask",mask)

		cv2.waitKey(1)
		print("Mask has been uploaded sucessfully")

	#cv2.imshow("mask", mask)
	#cv2.waitKey(1)

	num_seg = input("What is the size of the matrix (# of segment)? ")
	num_trials = input("How many data points do you want to collect for each segment? ")

	Dtr=[]
	#coord=[]  #map of coordinate for every segment ID in range (0,num_seg)
	#acquiring steady state data
	for j in range(0,int(num_trials)):
		print("acquiring data for SS Trial#%i" %j)
		while True:
			ret, frame = cap.read()
			frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
			(frame, gray, pix_cent_gray,pix_cent_bgr, mean_intensity, pixels_bgr,pixels_gray)=feature_extraction(frame, com_mask,size)
			reply=yes_or_no("Do you want to save this data?")
			if reply==True:		
				print("saving data for steady state, Trial#%i" %j)	
						#pdb.set_trace()
				new_feat=Feature(frame, gray, pix_cent_gray,pix_cent_bgr, mean_intensity, pixels_bgr,pixels_gray,"SS")
				Dtr.append(new_feat)
				break

	for i in range(0,int(num_seg)):
		print("\n\n####NEW SEGMENT %i####" %i)
		for j in range(0,int(num_trials)):

			print("\n\n### Press segment #%i Trial#%i ###"  %(i,j))

		    
			#reply=yes_or_no("Do you want to save this data?")
		    
			while True:

				input("Are you ready to acquired? Press Enter to continue...")
				print("acquiring data for segment#%i Trial#%i" %(i,j))
				ret, frame = cap.read()
				frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
				(frame, gray, pix_cent_gray,pix_cent_bgr, mean_intensity, pixels_bgr,pixels_gray)=feature_extraction(frame, com_mask,size)
				reply=yes_or_no("Do you want to save this data?")
				if reply==True:		
					print("saving data for segment#%i Trial#%i" %(i,j))	
					#pdb.set_trace()
					new_feat=Feature(frame, gray, pix_cent_gray,pix_cent_bgr, mean_intensity, pixels_bgr,pixels_gray,i)
					Dtr.append(new_feat)
					break

	print("\n\nData has been collected sucessfully")
	name_f = input("\nName your file: ")
	f=open(name_f+"_dtr.p","wb")
	pickle.dump(Dtr,f)
	print("Training data has been acquired sucessfully")
	f.close()

if __name__ == '__main__':
    #print("This only executes when %s is executed rather than imported" % __file__)
	main()