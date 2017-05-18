import pandas as pd


def c(*args):
    size = len(args)
    if  size == 1:
        return args[0]-1
    else:
        return [ int(i-1) for i in range(args[0], args[size-1]+1)]




# Importing the dataset
D = pd.read_csv('/home/rafsanjani/Machine_Learning/'
                'Machine_Learnin_A-Z_by_Kirill_E./'
                'Part_1:_Data_Preprocessing/'
                'Section 2 -------------------- '
                'Part 1 - Data '
                'Preprocessing --------------------/Data.csv')



D = D.drop_duplicates()     # Drop duplicates

X = D.iloc[:, c(1,3)].values
y = D.iloc[:,c(4)].values


# Handling the missing data
from sklearn.preprocessing import Imputer
X[:, [1,2]] = Imputer(missing_values='NaN', strategy='mean', axis=0).\
    fit_transform(X[:, [1,2]])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoding the independing variable
X[:,0] = LabelEncoder().fit_transform(X[:,0])

X = OneHotEncoder(categorical_features=[0]).fit_transform(X).toarray()

# Encoding the depending variable
y = LabelEncoder().fit_transform(y)

# Spliting 70% train, 30% test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

X_train = scale.fit_transform(X_train);
X_test = scale.fit_transform(X_test)


print(X_train); print()
print(X_test)


