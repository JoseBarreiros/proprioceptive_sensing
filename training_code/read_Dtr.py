###############################
# Extracting Trainning Data 
# Jose Barreiros, Alec Cornwell
# Jun, 2018
################################

import sys, cv2, argparse, time, pdb, pickle
import numpy as np 
from trainning_v4 import Feature 

def Read_Dtr(name_f): 

	f=open(name_f,"rb")
	Dtr=pickle.load(f)

	print("Succesfully loaded file")
	#print(Dtr)
	return Dtr