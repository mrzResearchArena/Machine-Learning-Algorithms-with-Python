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

# Calculate Error
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
                'Part_3:_ Classification/'
                'Section 14 - Logistic Regression/'
                'Logistic_Regression/Social_Network_Ads.csv')


X = D.iloc[:, [2,3]].values    # independent variable
y = D.iloc[:, c(5)].values   # dependent variable

X = X.astype(float)
y = y.astype(float)


# Spliting dataset 67% train & 33% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)
# ------------------------------------------------ #


# Feature Scalling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()

X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)
# ---------------------------------- #


# Classifier Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# from imblearn.over_sampling import ADASYN, SMOTE
# from imblearn.over_sampling import SMOTE
# from imblearn.ensemble import BalanceCascade

# Select a model ...
model = GaussianNB()
model.fit(X_train, y_train)


# Evaluation Performance
y_artificial = model.predict(X_test)
# y_pro = model.predict_proba(X_test)
print(y_artificial)


from sklearn.metrics import accuracy_score, \
                            confusion_matrix,\
                            precision_recall_fscore_support,\
                            precision_score,\
                            recall_score, f1_score, fbeta_score,\
                            roc_curve


CM = confusion_matrix(y_true=y_test, y_pred=y_artificial)
print(CM)


print(accuracy_score(y_true=y_test, y_pred=y_artificial))


# tpr, fpr, _ = roc_curve(y_true=y_test, y_score=y_pro)


# # Visualising the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('--- Age ---')
# plt.ylabel('--- Estimated Salary ---')
# plt.legend()
# plt.show()



# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Classifier (Test set)')
plt.xlabel('--- Age ---')
plt.ylabel('--- Estimated Salary ---')
plt.legend()
plt.show()


