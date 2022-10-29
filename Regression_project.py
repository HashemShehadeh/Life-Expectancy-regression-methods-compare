# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 20:53:25 2022

@author: hashem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset

dataset=pd.read_csv("Original_LifeExpectancy.csv")
dataset.info()

X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

#taking care of missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan , strategy="mean" )
imputer.fit(X[:,:18])
X[:,0:18]=imputer.transform(X[:,0:18])
imputer.fit(np.reshape(y, (-1,1)))
y=imputer.transform(np.reshape(y, (-1,1)))


#Splitting the dataset into a train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)



from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train,y_train)
y_pred= lr.predict(X_test)


from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)*100


#*******************************************************************************************



#to treat with the constatnt value
X=np.insert(X,0,1,axis=1)
#applying the evaluation method (Backward Elimination)
import statsmodels.api as sm
X_opt=np.array(X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]],dtype=float)
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()
    
#applying the evaluation method (Backward Elimination)
import statsmodels.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_ols=sm.OLS(endog=y,exog=x).fit()
        maxVar = max(regressor_ols.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_ols.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    
    regressor_ols.summary()
    return x


SL = 0.05
X_opt = X[:, :]
X_Modeled = backwardElimination(X_opt, SL)

regressor_ols=sm.OLS(endog=y,exog=X_Modeled).fit()
regressor_ols.summary()


from sklearn.model_selection import train_test_split
X_train_opt,X_test_opt,y_train,y_test= train_test_split(X_Modeled,y, test_size= 0.2 , random_state=0)



from sklearn.linear_model import LinearRegression
lr2= LinearRegression()
lr2.fit(X_train_opt,y_train)
y_pred2= lr2.predict(X_test_opt)

from sklearn.metrics import r2_score
score2 = r2_score(y_test,y_pred2)*100

        
                         
                             
                             
                             