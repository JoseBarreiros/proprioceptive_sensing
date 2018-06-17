###############################
# Trainning with Support Vector Machine
# Jose Barreiros, Alec Cornwell
# Jun, 2018
################################

import sys, cv2, argparse, time, pdb, pickle, random
import numpy as np 
from trainning_v4 import Feature 
from read_Dtr import Read_Dtr
from MergedData import merge_data
from sklearn import svm
from sklearn.grid_search import GridSearchCV

def get_labeled_data(Dtr):
	#pdb.set_trace()
	X=[]
	Y=[]
	for i in range(0,len(Dtr)):
		X.append(Dtr[i].mean_intensity[:,3])
		Y.append(Dtr[i].segmentID)

	X=np.asarray(X)
	Y=np.asarray(Y)
	

	return (X,Y)

def yes_or_no(question):
	while "the answer is invalid":
		reply = str(input(question+' (y/n): ')).lower().strip()
		if reply[0] == 'y':
			return True
		if reply[0] == 'n':
			return False

def randomize_order(X,Y):
	assert len(X) == len(Y)
	p = np.random.permutation(len(X))
	return(X[p],Y[p])

def split_data(X,Y,percentage_tr):
	#pdb.set_trace()
	size=len(X)
	size_tr=int(size*percentage_tr)

	ran_index = np.random.permutation(size)
	tr_index=ran_index[:size_tr]
	te_index=ran_index[size_tr:]

	Xtr=X[tr_index]
	Ytr=Y[tr_index]

   
	Xte=X[te_index]
	Yte=Y[te_index]

	return (Xtr,Ytr,Xte,Yte)

def main():
	#For single file
	name_f = input("What file do you want to open: ")
	#name_f="jun14_330pm_dtr.p"
	Dtr=Read_Dtr(name_f)

	#For merging files
	#name_f_D1="jun14_330pm_dtr.p"
	#name_f_D2="ss_dtr.p"
	#Dtr=merge_data(name_f_D1,name_f_D2)



	(X,Y)=get_labeled_data(Dtr)
	
	(X,Y)=randomize_order(X,Y)
	(Xtr,Ytr,Xte,Yte)=split_data(X,Y,0.7)
	
	clf=svm.SVC()
	#param_grid = { 
 	#	"kernel" : ['linear', 'rbf'],
	#	"gamma" : [1e-1, 1e-2],
	#	"C" : [1, 10, 100]}

	#clf=GridSearchCV(svm.SVC(), param_grid, cv=10, scoring='accuracy')
	clf.fit(Xtr,Ytr)
	print("SVM Model succesfully acquired")
	print(clf)
	#clf.best_params_
	#print(clf.best_params_)
	reply1=yes_or_no("Do you want to save this ML model?")
	if reply1==True:
		name_f = input("\nName your ML Model file: ") 
		f=open(name_f+".model","wb")         
		pickle.dump(clf,f)
		pickle.dump(clf,f)
		f.close()



	Ytr_predicted=clf.predict(Xtr)
	#pdb.set_trace()
	acc=(Ytr.astype('str')==Ytr_predicted).sum()/len(Ytr)
	print("\nTrainning Acc: %f" %acc)

	Yte_predicted=clf.predict(Xte)
	acc=(Yte.astype('str')==Yte_predicted).sum()/len(Yte)
	print("\nTesting Acc: %f" %acc)	
    

	#pdb.set_trace()
	reply=yes_or_no("Do you validate the model?")
	if reply==True:
		name_f = input("What is your validation file?: ")
		#name_f="jun15_1pm_val_dtr .p"
		Dva=Read_Dtr(name_f)

		#name_f_D1="val_jun14_6pm_dtr.p"
		#name_f_D2="ss_val_dtr.p"
		#Dva=merge_data(name_f_D1,name_f_D2)

		(Xva,Yva)=get_labeled_data(Dva)
		(Xva,Yva)=randomize_order(Xva,Yva)
		Yva_predicted=clf.predict(Xva)
		acc=(Yva.astype('str')==Yva_predicted).sum()/len(Yva)
		print("\nValidation Acc: %f" %acc)
		#pdb.set_trace()





if __name__ == '__main__':
    #print("This only executes when %s is executed rather than imported" % __file__)
	main()