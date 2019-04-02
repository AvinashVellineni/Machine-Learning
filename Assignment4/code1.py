import pandas as pd
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import math
import matplotlib.pyplot as plt


print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")
alpha_list = [pow(10,-7),pow(10,-6),pow(10,-5),pow(10,-4),pow(10,-3),pow(10,-2),pow(10,-1),pow(10,0),pow(10,1),pow(10,2),pow(10,3),
              pow(10,4),pow(10,5),pow(10,6),pow(10,7)]

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
        w=np.array(clf.intercept_)
        w1=np.array(clf.coef_)
        #print(w1)
        count4=0;
        if(w==0.0):
            count4=count4+1
        
        for l in range(0,len(w1[0])):
            if(w1[0][l]==0.0):
                count4=count4+1
        l2_num_zero_weights[i][j]=count4
        penalty_l2=np.sum(np.array(clf.intercept_)*np.array(clf.intercept_))  + np.sum(np.array(clf.coef_)*np.array(clf.coef_))
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

for i in range(len(Xs)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(l2_model_complexity[i],l2_train_cll[i], marker='o',label='train_cll')
    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('train_cll/test_cll')
    ax.plot(l2_model_complexity[i],l2_test_cll[i], marker='o',label='test_cll')
    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('train_cll/test_cll')
    ax.legend()
    j=i+1
    fig.suptitle('Dataset %d'%j)
    fig.tight_layout()

for i in range(len(Xs)):
    X_train, X_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1./3, random_state=int("6657"))
    #print(X_train)
    for j in range(len(alpha_list)):
        clf_l1=LogisticRegression(penalty="l1",C=alpha_list[j],random_state=42)       
        clf_l1.fit(X_train, y_train)
        w2=np.array(clf_l1.intercept_)
        w3=np.array(clf_l1.coef_)
        count5=0
        if(w2==0.0):
            count5=count5+1
        
        for l in range(0,len(w3[0])):
            if(w3[0][l]==0.0):
                count5=count5+1
            
            
        l1_num_zero_weights[i][j]=count5

for i in range(len(Xs)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list1,l2_num_zero_weights[i], marker='o',label='l2_num_zero_weights')
    ax.set_xlabel('Exponent of alpha values')
    ax.set_ylabel('l2_num_zero_weights/l1_num_zero_weights')
    ax.plot(list1,l1_num_zero_weights[i], marker='o',label='l1_num_zero_weights')
    ax.set_xlabel('Exponent of alpha values')
    ax.set_ylabel('l2_num_zero_weights/l1_num_zero_weights')
    ax.legend()
    j=i+1
    fig.suptitle('Dataset %d'%j)
    fig.tight_layout()

    

print("Model complexity l2")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l2_model_complexity[i]))


print("\nTrain CLL")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l2_train_cll[i]))

print("\nTest CLL")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l2_test_cll[i]))

print("\nl1_num_zero_weights")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l1_num_zero_weights[i]))

print("\nl2_num_zero_weights")
for i in range(10):
    print("\t".join("{0:.4f}".format(n) for n in l2_num_zero_weights[i]))

pickle.dump((l2_model_complexity, l2_train_cll,l2_test_cll,l2_num_zero_weights,l1_num_zero_weights), open('result.pkl', 'wb'))
