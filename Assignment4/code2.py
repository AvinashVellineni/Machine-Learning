# Programming assignment 2
# Prereqs: all previous prereqs, plus pandas
# Implement what is asked at the TODO section.
# You can import additional methods/classes if you need them.
import pandas as pd
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import math

print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")
alpha_list = [pow(10,-7),pow(10,-6),pow(10,-5),pow(10,-4),pow(10,-3),pow(10,-2),pow(10,-1),pow(10,0),pow(10,1),pow(10,2),pow(10,3),
              pow(10,4),pow(10,5),pow(10,6),pow(10,7)]

#alpha_list=[pow(10,-7)]

#for i in range(len(alpha_list)):
    #print(alpha_list[i])

#print(Xs)

#sum_2=0
#print(ys)
l2_train_cll = np.zeros((10, 15))
l2_test_cll  = np.zeros((10, 15))
l2_model_complexity = np.zeros((10, 15))
l2_num_zero_weights = np.zeros((10, 15))
l1_num_zero_weights = np.zeros((10, 15))
# Anumber A20406657
for i in range(len(Xs)):
    X_train, X_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1./3, random_state=int("6657"))
    #print(X_train)
    for j in range(len(alpha_list)):
        clf=LogisticRegression(penalty="l2",C=alpha_list[j],random_state=42)       
        clf.fit(X_train, y_train)
        #print(clf.coef_)
        #print(clf.intercept_)
        w=np.array(clf.intercept_)
        w1=np.array(clf.coef_)
        #print(w1)
##        penalty=np.sum(w*w)
##        penalty1=np.sum(w1*w1)
        penalty_l2=np.sum(np.array(clf.intercept_)*np.array(clf.intercept_))  + np.sum(np.array(clf.coef_)*np.array(clf.coef_))
        #print(penalty)
        #print(penalty1)
        #print(penalty_l2)
        sum_1=0
        sum_2=0
        n_zeros=0
        l2_model_complexity[i][j]=penalty_l2
        CLL_X_train=clf.predict_log_proba(X_train)
        CLL_X_test=clf.predict_log_proba(X_test)
        for k in range(0,len(CLL_X_train)):
            if y_train[k]==True:
                sum_1+=CLL_X_train[k][1]
            else:
                sum_1+=CLL_X_train[k][0]
        for m in range(0,len(CLL_X_test)):
            if y_test[m]==True:
                sum_2+=CLL_X_test[m][1]
            else:
                sum_2+=CLL_X_test[m][0]
                
        l2_train_cll[i][j]=sum_1
        l2_test_cll[i][j]=sum_2
        
            #print(y_train[k])
##        sum_1=0
##        sum_2=0
##        clf = BernoulliNB(alpha=alpha_list[j], binarize=0.0, class_prior=None, fit_prior=True)
##        clf.fit(X_train, y_train)
##        joint_X_train=clf._joint_log_likelihood(X_train)
##        joint_X_test=clf._joint_log_likelihood(X_test)
##        for k in range(0,len(joint_X_train)):
##            if y_train[k]==True:
##                sum_1+=joint_X_train[k][1]
##            else:
##                sum_1+=joint_X_train[k][0]
##            #print(y_train[k])
##        for m in range(0,len(joint_X_test)):
##            if y_test[m]==True:
##                sum_2+=joint_X_test[m][1]
##            else:
##                sum_2+=joint_X_test[m][0]
##        
##        train_jll[i][j]=sum_1
##        test_jll[i][j]=sum_2


            

    

        

        
        
        #print(sum_1)
            



        
        #clf_predict_train=clf.predict_log_proba(X_train)
        #clf_predict_test=clf.predict_log_proba(X_test)

        
        #print("Train predictions")
        #lenfth1=len(clf_predict_train)
        #for p in clf_predict_train
        
        
        
        #print(clf_predict_train)
        #print("test predictions")
        #print(clf_predict_test)
        #for k in range(len(clf_predict_train)):
            
            
            #sum_1=sum_1+clf_predict_train[k][0]
            #sum_2=sum_2+clf_predict_train[k][1]
        
        #print(sum_1)
        #print(sum_2)
        
        

                                 
        #train_jll=
        #test_jll=
        #predict_log_proba
        

    
    



## DO NOT MODIFY BELOW THIS LINE.

print("Model complexity l2")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in l2_model_complexity[i]))
	

print("\nTrain CLL")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in l2_train_cll[i]))
##
print("\nTest CLL")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in l2_test_cll[i]))

# Once you run the code, it will generate a 'results.pkl' file. Do not modify the following code.
#pickle.dump((train_jll, test_jll), open('result.pkl', 'wb'))
