# -*- coding: utf-8 -*-
"""
File:   RBF.py
Author:   Tashfique Hasnine Choudhury  
Date:   09.20.2021
Desc:   Implementation of RBF for Curve fitting
    
"""


""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib.pyplot as plt

plt.close('all') #close any open plots


""" ======================  Function definitions ========================== """

def plotData(x1,t1,x2=None,t2=None,x3=None,t3=None,x4=None,t4=None ,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    plt.figure(figsize=(8,6), dpi=80)
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    if(x2 is not None):
        p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot test data for evenly spaced mu
    if(x4 is not None):
        p4 = plt.plot(x4, t4, 'k') #plot test data for random spaced mu
    #add title, legend and axes labels
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    if(x2 is None):
        plt.legend((p1[0]),legend)
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    if(x4 is None):
        plt.legend((p1[0],p2[0],p3[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0],p4[0]),legend)
      
    plt.title(f's={s}, M={M}')
    #plt.savefig(f'Functions and estimates s={s}, M={M}.png')
    plt.show()                
 
def ErrorPlot(M, s):    
    """calculate for mean absolute error of test data - Evenly spaced mu"""
    xaxis=np.zeros(M)
    err=np.zeros(M)
    for i in range(M):
        p=rbf(x2, s=s, M=i+1, even_dis=True)
        err[i]=np.mean(abs(p-t2))
        xaxis[i]=i+1
    
    """calculate for mean absolute error of test data showing error bars for 10 runs - Randomly spaced mu"""
    Error_array=[]
    for p in range(10):
        er=np.zeros(M)
        for i in range(M):
            p=rbf(x2, s=s, M=i+1, even_dis=False)
            er[i]=np.mean(abs(p-t2))
        Error_array.append(er) #list of 10 mean error data of test data appended for each M
    
    Error_array=np.array(Error_array) #list turned into a 10xM array
    
    mean=[]
    for i in range(M):
        summ=0
        for j in range(10):
            summ+=Error_array[j][i]
        summ=summ/10
        mean.append(summ) #mean of each column of the Error_array
    arr = Error_array.T #Transpose to bring all 10 error values for a single M to a row
    plt.figure(figsize=(8,6), dpi=80)
    for i in range(M):
        plt.plot(np.ones(10)+i,(arr[i]), label=f'Error Bar for M={i+1}') #plot error bars for each M
    plt.plot(xaxis, mean, color='y', label='Mean Absolute error \nfor randomly spaced mu')
    plt.plot(xaxis, err, color='k', label='Mean Absolute error \nfor evenly spaced mu')
    plt.title(f's={s}, M={M}')
    plt.xlabel('M')
    plt.ylabel('Absolute Error')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    #plt.tight_layout()
    #plt.savefig(f'Abs Err s={s}, M={M}.png')
    plt.show()       

""" ======================  Variable Declaration ========================== """
#M = 5 #regression model order
#s = 0.5

""" =======================  Load Training Data ======================= """
data_uniform = np.load('train_data.npy').T
x1 = data_uniform[:,0]
t1 = data_uniform[:,1]

    
""" ========================  Train the Model ============================= """
def fitdata(xa,ta,M):
    Xa = np.array([xa**m for m in range(M+1)]).T
    wa = np.linalg.inv(Xa.T@Xa)@Xa.T@ta
    return wa

def rbf(x, s, M, even_dis=True): 
    if even_dis:
        mu = np.linspace(min(x1),max(x1),M) #take equally spaced points from train data
    else:
        mu = np.random.choice(x1, M, replace=False) #take random points from train data
    Phi = np.zeros((len(x1), M))
    Phi_T = np.zeros((len(x), M))
    for i in range(M):
            Phi[:,i] = np.exp(-0.5*((x1-mu[i])/s)**2)  #compute phi for train data
            Phi_T[:,i] = np.exp(-0.5*((x-mu[i])/s)**2)  #compute phi for test data
    Phi = np.concatenate((np.ones(len(x1))[:, np.newaxis], Phi), axis=1) #append column of ones to phi of train data
    Phi_T = np.concatenate((np.ones(len(x))[:, np.newaxis], Phi_T), axis=1) #append column of ones to phi of test data
    W = np.linalg.inv(Phi.T@Phi+(0.001*np.identity(M+1)))@Phi.T@t1 #Calculate weights from train data
    Ey = Phi_T@W #Estimate y
    return Ey
""" ======================== Load Test Data  and Test the Model =========================== """

x2 = np.load('test_data.npy').T

"""calculating true function"""
xrange = np.arange(-4,4,0.001) #get equally spaced points in the xrange
xrange = xrange[(xrange<-1.5)|(xrange>-0.5)] #clip the range to avoid undefined values
t2 = (xrange/(xrange+1)) #compute the true function value   

"""Prediction for both evenly and randomly distributed mu on test data"""
#p1 = rbf(x2, s, M, even_dis=True) #predictions for test data using weights computed from evenly spaced mu
#p2 = rbf(x2, s, M, even_dis=False) #predictions for test data using weights computed from randomly spaced mu

"""Prediction by polynomial curve fit"""
wa = fitdata(x1,t1,M=5)
xrang = np.arange(-4,4,0.001) #get equally spaced points in the xrange
xrang = xrang[(xrang<-1.5)|(xrang>-0.5)]
xa = np.array([xrang**m for m in range(wa.size)]).T  #extract the same features on test data
esty = xa@wa #compute the predicted value


 
""" ========================  Plot Results ============================== """

"""plots for same M different s"""
for s in [0.001, 0.01, 0.1, 0.5, 2, 10]:
    M=5
    p1 = rbf(x2, s=s, M=M, even_dis=True) #predictions for test data using weights computed from evenly spaced mu
    p2 = rbf(x2, s=s, M=M, even_dis=False) #predictions for test data using weights computed from randomly spaced mu
    plotData(x1,t1,xrange,t2,x2,p1,x2,p2,['Training Data', 'True Function', 'Evenly spaced RBF Estimate','Random RBF Estimate'])
    ErrorPlot(M=M, s=s)

"""plots for same s different M"""    
for M in [2,6,9,12,16,19]:
    s=0.5
    p1 = rbf(x2, s=s, M=M, even_dis=True) #predictions for test data using weights computed from evenly spaced mu
    p2 = rbf(x2, s=s, M=M, even_dis=False) #predictions for test data using weights computed from randomly spaced mu
    plotData(x1,t1,xrange,t2,x2,p1,x2,p2,['Training Data', 'True Function', 'Evenly spaced RBF Estimate','Random RBF Estimate'])
    ErrorPlot(M=M, s=s)


"""Additional tests conducted"""
#plt.figure(figsize=(8,6), dpi=80)
#plt.title(f's={s}, M={M}')
#plt.scatter(x1,t1, color='r', label='Train Data')
#plt.plot(xrang, esty, color='b', label='Polynomial Basis')
#plt.plot(x2,p1, color='g', label='Evenly spaced RBF')
#plt.plot(x2,p2, color='y', label='Randomly spaced RBF')
#plt.legend()
#plt.show()


#err1=np.mean(abs(p1-t2))   #Error for Evenly spaced RBF
#err2=np.mean(abs(p2-t2))   #Error for Randomly spaced RBF
#err3=np.mean(abs(esty-t2))   #Error for Polynomial Basis
#print('Error for Evenly spaced RBF=',err1)
#print('Error for Randomly spaced RBF=',err2)
#print('Error for Polynomial Basis=',err3)


#plt.figure(figsize=(8,6), dpi=80)
#plt.ylabel('t') #label x and y axes
#plt.xlabel('x')
#t3=(x2/(x2+1))
#plt.scatter(x1, t1, color='r', label='train data')
#plt.scatter(x2, t3, color='b', label='test data')
#plt.legend()