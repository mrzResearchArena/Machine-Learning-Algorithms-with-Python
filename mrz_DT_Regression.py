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
                'Part_2:_Regression/Section 6 - Polynomial'
                ' Regression/Polynomial_Regression/'
                'Position_Salaries.csv')



X = D.iloc[:, 1:2].values    # independent variable
y = D.iloc[:, c(3)].values   # dependent variable

# Feature Scalling
'''
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_y = StandardScaler()
X = scale_X.fit_transform(X)
y = scale_y.fit_transform(y)
'''

# Algorithms
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)
#linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

'''
y_artificial = scale_y.inverse_transform(
        model.predict(scale_X.transform(np.array([[6.5]]))))
'''

y_artificial = model.predict(6.5)


# Visualization with better resulation
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color='red')
plt.plot(X_grid, model.predict((X_grid)), color='blue')
plt.title('Truth or Bluff')
plt.show()



