# Basic library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
### ------------------------ ###



# Preprocessing library
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
### ------------------------ ###



# This function will work for 1-base index (Though Python, Java, C/C++ are 0-based index)
def c(*args):
    size = len(args)
    if  size == 1:
        return args[0]-1
    else:
        return [ int(i-1) for i in range(args[0], args[size-1]+1) ]
    ### Return


# Label Encoding
def labelEncoding(X, featureNumber):
    X[:, c(featureNumber)] = LabelEncoder().fit_transform(X[:, c(featureNumber)])
    return X
    ### Return


# One Hot Encoding
def oneHotEncoding(X, featureNumber):
    X = OneHotEncoder(categorical_features=[c(featureNumber)]).fit_transform(X).toarray()
    return X
    ### Return


def calculateError(y_artificial, y_test):
    C=0.0

    for i,j in zip(y_test, y_artificial):
        # print(j, i)
        C = C + ((i-j)**2.0)


    return (C/(y_test.shape[0]-2))**0.5


### ********************************************************************* ###



# Importing the dataset
D = pd.read_csv('/home/rafsanjani/Machine_Learning/'
                'Machine_Learnin_A-Z_by_Kirill_E./'
                'Part_2:_Regression/'
                'Section 5 - Multiple Linear Regression/'
                'Homework_Solutions/50_Startups.csv')



X = D.iloc[:, :-1].values # independent variable
y = D.iloc[:, c(5)].values # dependent variable



# Preprocessing

# Labeling for independent feature
X = labelEncoding(X, 4)     # 1-based index
X = oneHotEncoding(X, 4)    # 1-based index

# Labeling for dependent feature
# print(y)
# y = LabelEncoder().fit_transform(y)

# For numerical values in y = LabelEncoder().fit_transform(y)
"""
It first sort the all values, then
Let, values for y = [ 23.71   10.71   23.71   23.71   23.72  231.71   33.71 ]
After sorting value looks like = [10.71   23.71   23.71   23.71   23.72   33.71   231.71 ]
assign one by one = [ 0 1 1 1 2 3 4 ]
finally looks like = [ 1 0 1 1 2 4 3 ]
"""


# For nominal values in y = LabelEncoder().fit_transform(y)
"""
# It first sort the all values, then
# Let, values for y = ['Canada' 'USA' 'Bangladesh' 'Armenia' 'Canada' 'Canada' 'Zimbabewe']
# After sorting value looks like = [ 'Armenia' 'Bangladesh' 'Canada' 'Canada' 'USA' 'Zimbabewe' ] # dictionary order
# assign one by one = [ 0 1 2 2 3 4 ]
# finally looks like = [ 2 3 1 0 2 2 4 ]
"""


# Avoiding dummy variable
X = X[:,1:]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=0)

"""
print(X_train); print()
print(X_test); print()
print(y_train); print()
print(y_test); print()
"""

# train model
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)

y_artificial = model.predict(X_test)

# print(y_artificial); print()
# print(y_test)

# print(calculateError(y_artificial, y_test))

import statsmodels.formula.api as sm

X = np.append(arr=np.ones([50,1]), values=X, axis=1)

### --- for Backword elimination --- ### 
X_opt = X[:, [0,1,2,3,4,5]]
model_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(model_OLS.summary())
print('-------------------------------')

X_opt = X[:, [0,1,3,4,5]]
model_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(model_OLS.summary())
print('-------------------------------')

X_opt = X[:, [0,3,4,5]]
model_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(model_OLS.summary())
print('-------------------------------')


X_opt = X[:, [0,3,5]]
model_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(model_OLS.summary())
print('-------------------------------')


X_opt = X[:, [0,3]]
model_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(model_OLS.summary())
print('-------------------------------')




