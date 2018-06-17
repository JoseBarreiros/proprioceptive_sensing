###############################
# Appending Trainning Data
# Jose Barreiros, Alec Cornwell
# Jun, 2018
################################

import sys, cv2, argparse, time, pdb, pickle, random
import numpy as np 
from trainning_v4 import Feature 
from read_Dtr import Read_Dtr
from sklearn import svm

def merge_data(file_D1,file_D2):
	#pdb.set_trace()
	
	D1=Read_Dtr(file_D1)
	D2=Read_Dtr(file_D2)
	D=D1
	D.extend(D2)
	return D


