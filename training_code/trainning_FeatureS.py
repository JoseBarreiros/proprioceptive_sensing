###############################
# Trainning with Support Vector Machine
# Jose Barreiros, Alec Cornwell
# Jun, 2018
################################

import sys, cv2, argparse, time, pdb, pickle, random, os
import numpy as np 
from trainning_v4 import Feature 
from read_Dtr import Read_Dtr
from MergedData import merge_data
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier 

def get_labeled_data(Dtr):
	#pdb.set_trace()
	X=[]
	Y=[]
	for i in range(0,len(Dtr)):
		X.append(Dtr[i].mean_intensity)
		
		#gray pixels
		#for j in range(0,32):
		#	xx=Dtr[i].pixels_gray[j][:5900]

		#BGR PIXELS
		#for j in range(0,32):
		#	xx=Dtr[i].pixels_bgr[j][:5900]
			##bg=Dtr[i].pixels_bgr[j][:5900][:,0].astype(float)*Dtr[i].pixels_bgr[j][:5900][:,1].astype(float)
			##r=Dtr[i].pixels_bgr[j][:5900][:,2].astype(float)
			##xx=np.hstack((np.array([bg]).T,np.array([r]).T))
		#	xx=np.ravel(xx)
		
		#for pix center bgr
		#X.append(Dtr[i].pix_cent_bgr.ravel())

		#for frame rgb
		#pdb.set_trace()
		#X.append(cv2.resize(Dtr[i].frame, (0,0), fx=0.5, fy=0.5).flatten())
		#X.append(Dtr[i].frame.flatten())
		#pdb.set_trace()
		

		#X.append(xx)

		#X.append(Dtr[i].pixels_gray)
		Y.append(Dtr[i].segmentID)
	#pdb.set_trace()
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
	#pdb.set_trace()
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

def Read_Dtr_folder(folder):
	#pdb.set_trace()
	Dtr=[]

	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		if filename.endswith(".p"):
			path=os.path.join(folder, filename)
			print(path)

			Dtemp=Read_Dtr(path)
			Dtr.extend(Dtemp)

			continue
		else:
			continue
	return Dtr

def feature_ratio(X):
	#pdb.set_trace()
	N=X.shape[0]
	M=X.shape[1]
	XX=np.zeros((N,M*M))
	for i in range(0,N):
		X_temp=[]
		#pdb.set_trace()
		for j in range(0,M):
			for k in range(0,M):
				X_temp.append(X[i][j]-X[i][k])
		XX[i]=np.array(X_temp)
	return XX



def main():
	#For single file
	#name_f = input("What file do you want to open: ")

	#name_f="jul24_520_coat_dtr.p"
	#Dtr=Read_Dtr(name_f)

	#for a folder
	name_folder="jul26_350"
	Dtr=Read_Dtr_folder(name_folder)

	#pdb.set_trace()
	

	#For merging files
	#name_f_D1="jun16_540pm_dark_dtr.p"
	#name_f_D2="jul19_6pm_dark_dtr.p"
	#Dtr=merge_data(name_f_D1,name_f_D2)



	(X,Y)=get_labeled_data(Dtr)
	#pdb.set_trace()
	(X,Y)=randomize_order(X,Y)
	#(Xtr,Ytr,Xte,Yte)=split_data(X,Y,0.6)
	
	(Xtr,Ytr,X_temp,Y_temp)=split_data(X,Y,0.6)

	(Xte,Yte,Xva,Yva)=split_data(X_temp,Y_temp,0.6)
	Xtr=feature_ratio(Xtr)
	Xte=feature_ratio(Xte)
	Xva=feature_ratio(Xva)
	#pdb.set_trace()
	
	#Model Selection

	#clf=svm.SVC()
	clf=KNeighborsClassifier()
	#clf=Perceptron(penalty=None, fit_intercept=True, max_iter=100000, tol=None, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False, n_iter=None)
	#clf = tree.DecisionTreeClassifier()
	#clf = GaussianNB()
	#clf=MultinomialNB()
	#clf=MLPClassifier()
#clf=svm.SVC(C=10,gamma=0.01,kernel='rbf')

	#param_grid = { 
 	#	"kernel" : ['rbf','linear'],
	#	"gamma" : [1,1e-1, 1e-2, 1e-3],
	#	"C" : [0.1,1,10,100]}
#	param_grid = { 
# 		"kernel" : ['rbf'],
#		"gamma" : [1],
#		"C" : [0.1,1]}
	#param_grid={'n_neighbors': [1,2,3,4,5, 6, 7, 8, 9], 'metric': ['minkowski', 'euclidean', 'manhattan'], 'weights': ['uniform', 'distance']}
	param_grid={'n_neighbors': [6], 'metric': ['minkowski'], 'weights': ['distance']}
	#param_grid={"alpha":[0.0001]}
	#param_grid = {'criterion':["gini","entropy"],'max_depth': [3,4,5,6,7,8,9,10],'min_impurity_decrease':[0,0.1,0.2]}
	#param_grid={'priors':[None]}
	#param_grid={'alpha':[0,0.5,0.7,1]}
	
	#param_grid={'hidden_layer_sizes':[(4),(10),(20),(100),(10,10),(20,20),(10,20),(20,10),(15,20),(7,3,5),(20,100,20),(50,10,50),(10,20,50),(20,15,10),(5,17,37),(10,10,10,10),(30,30,10,40),(10,50,50,10),(100,100,100,100),(100,10,100), (100, 100, 100)],'activation':["identity","logistic","tanh","relu"],'solver':["lbfgs","adam"]}
	#param_grid={'hidden_layer_sizes':[(20,100,20)],'activation':["relu"],'solver':["lbfgs"]}
	
	clf=GridSearchCV(clf, param_grid, cv=2, scoring='accuracy')




	clf.fit(Xtr,Ytr)
	print("Model succesfully acquired")
	print("Best score:")
	print(clf.best_score_)
	print("Best parameters:")
	print(clf.best_estimator_)


	
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
    

	Yva_predicted=clf.predict(Xva)
	acc=(Yva.astype('str')==Yva_predicted).sum()/len(Yva)
	print("\nValidation Acc: %f" %acc)	



	#pdb.set_trace()
	reply=yes_or_no("Do you validate the model?")
	if reply==True:
		#name_f = input("What is your validation file?: ")
		
		#for single file
		#name_f="jun16_6pm_val_dark_dtr.p"
		#Dva=Read_Dtr(name_f)

		#for folder
		name_folder="jul26_val"
		Dva=Read_Dtr_folder(name_folder)

		#name_f_D1="val_jun14_6pm_dtr.p"
		#name_f_D2="ss_val_dtr.p"
		#Dva=merge_data(name_f_D1,name_f_D2)

		(Xva_,Yva_)=get_labeled_data(Dva)


		(Xva_,Yva_)=randomize_order(Xva_,Yva_)
		#pdb.set_trace()
		Xva_=feature_ratio(Xva_)

		Yva_predicted_=clf.predict(Xva_)
		acc=(Yva_.astype('str')==Yva_predicted_).sum()/len(Yva_)
		print("\nValidation Acc: %f" %acc)
		#pdb.set_trace()





if __name__ == '__main__':
    #print("This only executes when %s is executed rather than imported" % __file__)
	main()