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



X = D.iloc[:, 1:2].values # independent variable
y = D.iloc[:, c(3)].values # dependent variable

from sklearn.linear_model import LinearRegression
modelLinear = LinearRegression().fit(X=X, y=y)
# y_artificial = modelLinear.predict(X)


from sklearn.preprocessing import PolynomialFeatures

polyRegression = PolynomialFeatures(degree=4)
X_poly = polyRegression.fit_transform(X)
polyRegression.fit(X_poly, y)

modelPoly = LinearRegression().fit(X_poly, y)



# Visualization Linear Model
plt.scatter(X, y, color='red')
plt.plot(X, modelLinear.predict(X), color='blue')
plt.title('Truth or Bluff')
plt.show()

# Visualization Polynomial Model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color='red')
plt.plot(X_grid, modelPoly.predict(polyRegression.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff')
plt.show()


#Prediction a new result based on linear model
print(modelLinear.predict(6.5))

#Prediction a new result based on polynomial model
print(modelPoly.predict(polyRegression.fit_transform(6.5)))




