# Avoiding warning
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
# ________________________________


# Essential Library
import pandas as pd
import numpy as np
# ________________________________


# scikit-learn for classifiers :
from sklearn.linear_model import LogisticRegression


# scikit-learn for performance measures :
from sklearn.metrics import accuracy_score,\
    confusion_matrix,\
    roc_auc_score,\
    average_precision_score,\
    f1_score,\
    matthews_corrcoef



# ////////////////////////////////////////////////////////////////////////////////////////////


# Step 01 : Load the dataset :
pima = '/home/mrz/Documents/pima.csv'
D = pd.read_csv(pima, header=None)
D = D.drop_duplicates()                 # Return : Unique records/instances.
# __________________________________________________________________________________



# Step 02 : Divided: features(X) and classes(y) from dataset(D).
X = D.iloc[:, :-1]
X = pd.get_dummies(X).values   # Convert categorical features into OneHotEncoder.
y = D.iloc[:, -1].values
# __________________________________________________________________________________



# Step 03 : Handle the missing values
from sklearn.preprocessing import Imputer
X[:, 0:X.shape[1]] = Imputer(strategy='mean').fit_transform(X[:, 0:X.shape[1]])
# We can use more stategy = 'median' or stategy = 'most_frequent'
# __________________________________________________________________________________



# Step 04 : Scaling the features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X = StandardScaler().fit_transform(X)



# Step 05 : Encoding y
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
# _________________________________________________________________________________



from sklearn.utils import shuffle
X, y = shuffle(X, y)  # Avoiding bias
# __________________________________________________________________________________



# Step 06 : Spliting with 10-FCV :
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)

Accuray = []
auROC = []
avePrecision = []
F1_Score = []
AUC = []
MCC = []
CM = np.array([
    [0, 0],
    [0, 0],
], dtype=int)

for train_index, test_index in cv.split(X, y):

    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

    model = LogisticRegression() # Change classifier

    model.fit(X_train, y_train)

    yHat= model.predict(X_test) # predicted labels

    y_proba = model.predict_proba(X_test)[:, 1]

    Accuray.append(accuracy_score(y_pred=yHat, y_true=y_test))
    auROC.append(roc_auc_score(y_test, y_proba))
    avePrecision.append(average_precision_score(y_test, y_proba))  # auPR
    F1_Score.append(f1_score(y_true=y_test, y_pred=yHat))
    MCC.append(matthews_corrcoef(y_true=y_test, y_pred=yHat))

    CM += confusion_matrix(y_pred=yHat, y_true=y_test)

print('Accuracy: {:.4f} ({:0.2f}%)'.format(np.mean(Accuray), np.mean(Accuray)*100.0))
print('auROC: {0:.4f}'.format(np.mean(auROC)))
print('auPR: {0:.4f}'.format(np.mean(avePrecision))) # average_Precision
print('F1-score: {0:.4f}'.format(np.mean(F1_Score)))
print('MCC: {0:.4f}'.format(np.mean(MCC)))

TN, FP, FN, TP = CM.ravel()
print('Sensitivity (+): {0:.4f}'.format( (TP) / (TP + FN)) )
print('Specificity (-): {0:.4f}'.format( (TN) / (TN + FP)) )
print('Confusion Matrix:')
print(CM)
print('___________________________')





