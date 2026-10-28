# Python For Data Science Cheat Sheet: Scikit-learn
###### Scikit-learn is an open source Python library that implements a range of machine learning, preprocessing, cross-validation and visualization algorithms using a unified interface.

## A Basic Example
```
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = dataset.load_iris()
X, y = iris.data[:, :2], iris.target
X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=33)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict (X_test)
accuracy_score(y_text, y_pred)
```

## Preprocessing The Data
### Standardization
###### Standardization transforms your data so that it has a mean of 0 and a standard deviation of 1. It preserves the original shape of the distribution but shifts and rescales the values.
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
```
### Normalization
###### Normalization rescales tha data so that all values are squished into a specific range, most commonly between 0 and 1.
```
from sklearn.preprocessing import Normalizer
scaler = Normalizer.fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
```
### Binarization
###### Binarization is the process of thresholding numerical values into binary choices (0 or 1). It turns a continuous variable into a hard boolean flag based on a specific threshold.
```
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)
```
### Encoding Categorical Features
###### Encoding Categorical Features converts text-based or discrete qualitative data into numrical values. 
> Machine Learning algorithms rely on mathematical equations and matrix multiplication. They cannot process raw text strings directly.
```
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
```
### Imputing Missing Values
###### Imputing Missing Values is the process of filling in blanks, null or missing data points with substituted statistical values rather than deleting the incomplete rows entirely.
```
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit_transform(X_train)
```
### Generating Polynomial Features
###### Generating Polynomial Features ia a feature engineering techniques that adds complexity to a model by creating new, non-linear combinations of existing features.
> It is a way to help machine learning models see curved relationships in data when they are only designed to see straight lines.
```
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(5)
poly.fit_transform(X)
```
### Training And Test Data
###### Training and testin data are the two essential subsets into which a dataset is split when building a machine learning model.
> Training data teaches the model how to look for patterns and make predictions, while testing data evaluates how accurately the model performs on new information.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```
