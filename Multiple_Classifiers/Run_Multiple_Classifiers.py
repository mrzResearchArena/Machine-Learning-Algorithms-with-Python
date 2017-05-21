# Avoiding warning
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
# _______________________________


# Essential Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# _______________________________

# Load dataset
D = pd.read_csv('/home/rafsanjani/Desktop/monodata.csv', header=None)
D = D.drop_duplicates() # Return : each row are unique value
# ___________________________________________________________________

# print(D.shape) # Return : row, column

# Divide features (X) and classes (y)
X = D.iloc[:,0:20].values
y = D.iloc[:,20].values
# ____________________________________


# Handle the missing values with "mean"
from sklearn.preprocessing import Imputer
X[:, [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]] = \
    Imputer().fit_transform(X[:, [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19]])
# __________________________________________________________________________________________________


# Spliting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=0.75, random_state=0)
# __________________________________________________________


# Features scalling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
# _____________________________________________________________


# Machine Learning Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
                             RandomForestClassifier,\
                             AdaBoostClassifier,\
                             GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

# _______________________________________________________________________


# Matrix
from sklearn.metrics import accuracy_score, \
                            log_loss, \
                            classification_report, \
                            confusion_matrix

from pandas_ml import ConfusionMatrix   # I'm using 'pandas_ml' for better confusion matrix than 'scikit-learn'.


classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(),
    SVC(kernel='rbf', probability=True),
    GaussianNB(),
    BaggingClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    XGBClassifier()
]



for classifier in classifiers:

    model = classifier.fit(X_train, y_train)

    y_artificial = model.predict(X_test) # Predicted

    name = classifier.__class__.__name__

    TN, FP, FN, TP = confusion_matrix(y_true=y_test, y_pred=y_artificial).ravel()

    print('_' * 43)
    print('Classifier : {}'.format(name))
    print('Accuracy : {0:.3f} %'.format(accuracy_score(y_true=y_test, y_pred=y_artificial)*100.0))
    # print('_'*40)

    print()
    print('Confusion Matrix :')
    CM = ConfusionMatrix(y_true = y_test, y_pred = y_artificial)
    print(CM)
    print()

    print('TN = {}'.format(TN))
    print('FP = {}'.format(FP))
    print('FN = {}'.format(FN))
    print('TP = {}'.format(TP))
    print()

    # CM.print_stats() # For Statistics based-on confusion matrix.

# _______________________________________________________________________________________________________


