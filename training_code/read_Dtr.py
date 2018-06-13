###############################
# Extracting Trainning Data 
# Jose Barreiros, Alec Cornwell
# Jun, 2018
################################

import sys, cv2, argparse, time, pdb, pickle
import numpy as np 
from trainning_v1 import Feature 

name_f = input("What file do you want to open: ")
f=open(name_f,"rb")
Dtr=pickle.load(f)

print("Succesfully loaded file")
print(Dtr)
pdb.set_trace()