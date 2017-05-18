import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def c(*args):
    size = len(args)
    if  size == 1:
        return args[0]-1
    else:
        return [ int(i-1) for i in range(args[0], args[size-1]+1)]



# Importing the dataset
D = pd.read_csv('/home/rafsanjani/Machine_Learning/'
                'Machine_Learnin_A-Z_by_Kirill_E./'
                'Part_2:_Regression/Section 4 - Simple Linear Regression/'
                'Salary_Data.csv')


X = D.iloc[:,0:1].values # dependent variable
y = D.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=.67, random_state=42)


# train model
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)


# prediction
y_artificial = model.predict(X_test)
print(y_artificial)


# Visualization Train
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')

plt.title('Salary vs. Exp')
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()


# Visualization Test
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, model.predict(X_test), color='blue')

plt.title('Salary vs. Exp')
plt.xlabel('Exp')
plt.ylabel('Salary')

plt.show()


