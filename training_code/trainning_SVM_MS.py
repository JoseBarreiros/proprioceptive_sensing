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
		#X.append(Dtr[i].mean_intensity)
		X.append(Dtr[i].pixels_bgr)
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
	#name_f = input("What file do you want to open: ")
	name_f="jul24_520_coat_dtr.p"
	Dtr=Read_Dtr(name_f)

	#For merging files
	#name_f_D1="jun14_330pm_dtr.p"
	#name_f_D2="ss_dtr.p"
	#Dtr=merge_data(name_f_D1,name_f_D2)



	(X,Y)=get_labeled_data(Dtr)
	
	(X,Y)=randomize_order(X,Y)
	(Xtr,Ytr,Xte,Yte)=split_data(X,Y,0.8)
	
	#Model Selection

	#clf=svm.SVC()
	#clf=KNeighborsClassifier()
	#clf=Perceptron(penalty=None, fit_intercept=True, max_iter=100000, tol=None, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False, n_iter=None)
	#clf = tree.DecisionTreeClassifier()
	#clf = GaussianNB()
	clf=MultinomialNB()

	#clf=MLPClassifier(max_iter=500)
	#param_grid = { 
 	#	"kernel" : ['rbf','linear'],
	#	"gamma" : [1,1e-1, 1e-2, 1e-3],
	#	"C" : [0.1,1,10,100]}
	#param_grid={'n_neighbors': [1,2,3,4,5, 6, 7, 8, 9], 'metric': ['minkowski', 'euclidean', 'manhattan'], 'weights': ['uniform', 'distance']}
	#param_grid={"alpha":[0.0001]}
	#param_grid = {'criterion':["gini","entropy"],'max_depth': [3,4,5,6,7,8,9,10],'min_impurity_decrease':[0,0.1,0.2]}
	#param_grid={'priors':[None]}
	param_grid={'alpha':[0,0.5,0.7,1]}
	#param_grid={'hidden_layer_sizes':[(4),(10),(20),(100),(10,10),(20,20),(10,20),(20,10),(15,20),(7,3,5),(20,100,20),(50,10,50),(10,20,50),(20,15,10),(5,17,37),(10,10,10,10),(30,30,10,40),(10,50,50,10),(100,100,100,100),(100,10,100), (100, 100, 100)],'activation':["identity","logistic","tanh","relu"],'solver':["lbfgs","adam"]}
	clf=GridSearchCV(clf, param_grid, cv=2, scoring='accuracy')




	#print(clf)
	#clf.best_params_
	#clf.get_params()
	clf.fit(Xtr,Ytr)
	print("SVM Model succesfully acquired")
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
    

	#pdb.set_trace()
	reply=yes_or_no("Do you validate the model?")
	if reply==True:
		#name_f = input("What is your validation file?: ")
		name_f="jun16_6pm_val_dark_dtr.p"
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