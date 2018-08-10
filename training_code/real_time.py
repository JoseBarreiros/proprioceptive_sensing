###############################
# Real time evaluation of ML Model for poked segment detection 
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
from trainning_v4 import Feature 
from trainning_v4 import feature_extraction 
from trainning_v4 import yes_or_no
import matplotlib.pyplot as plt
from collections import Counter


def main():
	print("##### Sequence for real time detection #####\n")
	print("##### for surface press location sensing in a 3D soft object #####\n")

	#Real time view of expanded surface of the cone
	img = plt.imread("coordinates_cone.png")
	fig, ax = plt.subplots()
	ax.imshow(img, extent=[0, 600, 0, 600])
	point,=ax.plot(87, 502, 'o', linewidth=5, markersize=22,color='firebrick',alpha=0.5)
	plt.axis('off')
	plt.pause(0.05)
	#dictiomary with coordinates
	coord_dict = {'SS' : (87,502), '0' : (236,459), '1' : (271,445), '2' : (313, 448), '3' : (353,456), '4' : (204,396), '5' : (268,377), '6' : (321,376), '7' : (380,391), '8' : (187,340), 
				  '9' : (254,318), '10' : (332,315), '11' : (408, 337), '12' : (152,282), '13' : (243, 258), '14' : (336, 256), '15' : (429, 286), '16' : (127, 232), '17' : (231, 197), '18' : (345, 197), 
				  '19' : (453, 227), '20' : (108, 173), '21' : (229, 141), '22' : (357, 139), '23' : (480, 179)}

	print("Opening Camera\n")
	cap = cv2.VideoCapture(0)
	#cap.set(3,270) #width
	#cap.set(4,480) #height
	#cap.set(5,1024) #fps

	plt.pause(0.05)
	

	time.sleep(1)

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
		#name_f = input("What file do you want to open as mask: ")
		name_f="jul24_420.mask"
		f=open(name_f,"rb")
		
		mask=pickle.load(f)
		com_mask=pickle.load(f)
		size=mask.shape
		#pdb.set_trace()
		cv2.imshow("loaded mask",mask)

		cv2.waitKey(1)
		print("Mask has been uploaded sucessfully")


	#name_f = input("What file do you want to open as ML model: ")
	#name_f="jun15_12pm.model"
	name_f="coating1.model"
	f=open(name_f,"rb")
	clf=pickle.load(f)
	while True:
		#pdb.set_trace()
		#t = time.time()
		#time.start

		ret, frame = cap.read()

		frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
		#cv2.imshow("current_frame",frame)
		#cv2.waitKey(1)

		(frame_, gray, pix_cent_gray,pix_cent_bgr, mean_intensity, pixels_bgr,pixels_gray)=feature_extraction(frame, com_mask,size)
		X=mean_intensity[:,3]
		#X=pix_cent_bgr.ravel()
		X=X.reshape(1,-1)
		#Y_predicted=clf.predict(X)

		Y_predicted1=clf.predict(X)
		Y_predicted2=clf.predict(X)
		Y_predicted3=clf.predict(X)
		Y_predicted4=clf.predict(X)
		ccc=Counter([Y_predicted1[0],Y_predicted2[0],Y_predicted3[0],Y_predicted4[0]])
		#pdb.set_trace()

		Y_predicted=ccc.most_common()[0][0]
		#Y_predicted='SS'
		
		print("Pressed segment: %s" %Y_predicted)
		#pdb.set_trace()
		(xx,yy)=coord_dict[Y_predicted]#retrieve from dictionary(int(Y_predicted))
		point.remove()
		plt.pause(0.05)
		point,=ax.plot(xx, yy, 'o', linewidth=5, markersize=22,color='firebrick',alpha=0.5)

		plt.pause(0.05)
		#time.sleep(0.05)
		#elapsed = time.time() - t
		#print("Elapsed_time: %s" %str(elapsed))
	plt.show()

if __name__ == '__main__':
    #print("This only executes when %s is executed rather than imported" % __file__)
	main()