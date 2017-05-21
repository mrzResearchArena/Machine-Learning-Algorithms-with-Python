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
D = D.drop_duplicates()  # Return : each row are unique value
# ____________________________________________________________________

# print(D.shape) # Return : row, column

# Divide features (X) and classes (y)
X = D.iloc[:, 0:20].values
y = D.iloc[:, 20].values
# ___________________________________

# Handle the missing values with "mean"
from sklearn.preprocessing import Imputer

X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]] = \
    Imputer().fit_transform(X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                  17, 18, 19]])
# __________________________________________________________________________________________


# Spliting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=0.75, random_state=0)

# _____________________________________________________________

# Features scalling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

# _________________________________________________


# Matrix
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_curve, auc

from pandas_ml import ConfusionMatrix  # I'm using 'pandas_ml' for better confusion matrix than 'scikit-learn'.
# ______________________________________________________________________________________________________________


# Machine Learning Classifiers
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit_transform(X_train, y_train)
y_artificial = model.predict(X_test)  # Predicted

# Evaluation

TN, FP, FN, TP = confusion_matrix(y_true=y_test, y_pred=y_artificial).ravel()
print('_' * 43)
print('Classifier : GradientBoostingClassifier')
print('Accuracy : {0:.3f} %'.format(accuracy_score(y_true=y_test, y_pred=y_artificial) * 100.0))
# print('_'*40)

print()
print('Confusion Matrix :')
CM = ConfusionMatrix(y_true=y_test, y_pred=y_artificial)
print(CM)
print()

print('TN = {}'.format(TN))
print('FP = {}'.format(FP))
print('FN = {}'.format(FN))
print('TP = {}'.format(TP))
print()

FPR, TPR, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])

roc_auc = auc(FPR, TPR)


# ROC Curve (using Python)
plt.figure()
plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# ROC Curve (using R)
# Compile it IPython notebook
from ggplot import *
df = pd.DataFrame(dict(fpr=FPR, tpr=TPR))
ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')



