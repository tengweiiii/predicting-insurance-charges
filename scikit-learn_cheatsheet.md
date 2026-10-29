# Python For Data Science Cheat Sheet: Scikit-learn
##### Scikit-learn is an open source Python library that implements a range of machine learning, preprocessing, cross-validation and visualization algorithms using a unified interface.

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
##### Standardization transforms your data so that it has a mean of 0 and a standard deviation of 1. It preserves the original shape of the distribution but shifts and rescales the values.
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
```
### Normalization
##### Normalization rescales tha data so that all values are squished into a specific range, most commonly between 0 and 1.
```
from sklearn.preprocessing import Normalizer
scaler = Normalizer.fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
```
### Binarization
##### Binarization is the process of thresholding numerical values into binary choices (0 or 1). It turns a continuous variable into a hard boolean flag based on a specific threshold.
```
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)
```
### Encoding Categorical Features
##### Encoding Categorical Features converts text-based or discrete qualitative data into numrical values. 
> Machine Learning algorithms rely on mathematical equations and matrix multiplication. They cannot process raw text strings directly.
```
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
```
### Imputing Missing Values
##### Imputing Missing Values is the process of filling in blanks, null or missing data points with substituted statistical values rather than deleting the incomplete rows entirely.
```
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit_transform(X_train)
```
### Generating Polynomial Features
##### Generating Polynomial Features ia a feature engineering techniques that adds complexity to a model by creating new, non-linear combinations of existing features.
> It is a way to help machine learning models see curved relationships in data when they are only designed to see straight lines.
```
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(5)
poly.fit_transform(X)
```
### Training And Test Data
##### Training and testin data are the two essential subsets into which a dataset is split when building a machine learning model.
> Training data teaches the model how to look for patterns and make predictions, while testing data evaluates how accurately the model performs on new information.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```
---
## Create Your Model
### Supervised Learning Estimators
##### Supervised learning estimators use labeled training data where input features are matched with known output targets to learn patterns and predict outcomes on new, unseen data.
#### Linear Regression
* **What it does:** Predicts a continuous, numerical value by drawing a straight line through data points
* **How it works:** It looks at the relationship between the input and target output and fits a line that minimizes the overall distance between the actual data points and the line itself
* **Real-world example**: Estimating a person'd weight based on their height or forecasting next month's sales revenue
```
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
```
#### Support Vector Machines (SVM)
* **What it does**: Finds the best wall or boundary (called a hyperplane) to separate different classes of data
* **How it works**: It tries to find a boundary that leaves the widest possible "street" or margin between the group. The data points closest to this boundary are called support vectors
* **Real-world example**: Classifying emails into spam vs. not spam or distinguishing benign tumors from malignant tumors in medical scans
```
from sklearn.svm import SVC
svc = SVC(kernel='linear')
```
#### Naive Bayes
* **What it does**: A fast classification method built on probability and Bayes' theorem
* **How it works**: It calculates the chance that a data point belongs to a certain category based on past examples
* **Real-world example**: Filtering junk mail or categorizing news articales into sports, politics or tech
> It is called 'Naive' because it makes a bold assumption. It treats every feature as completely independent of the others. For example, pretending the words "free", "winner", and "click" in an email have no relation to one another
```
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
```
#### K-Nearest Neighbors (KNN)
* **What it does**: Classifies a new data point or predicts a value by looking as its closest neighbors
* **How it works**: When a new, unknown data point arrives, KNN measures the distance (such as Euclidean distance) to the K closest points in the training set. If it is a classification task, whichever label wins the majority vote among those neighbors is assigned to the new point
* **Real-world example**: Recommendation systems (e.g., "people who bought similar items also bought this") or recognizing handwritten digits
```
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
```

### Unsupervised Learning Estimators
##### Unsupervised learning estimators work with unlabeled data, the computer is given raw data with no target answers, scores, or correct results. The algorithm must explore the data on its own to uncover hidden structures, groupings, or patterns
#### Principal Component Analysis (PCA)
* **What it does**: Reduces the number of features (variables) in a massive dataset while keeping as much of the original information as possible
* **How it works**: It finds the best angles (called principal components) that capture the maximum variance or "spread" of the data, allowing to drop less important angles and compress the dataset
* **Real-world example**: Compressing high-resolution images to save storage or simplifying genomics data with thousands of gene expressions down to a few key traits
```
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
```
#### K Means
* **What it does**: Groups unlabeled data points into a specific number (K) of distinct clusters based on their similarities
* **How it works**: It randomly places K invisible center points (centriods) in the data space. It then assigns each data point to its nearest center, calculates the actual middle of that new group, moves the center there and repeats the process until the group stop changing
* **Real-world example**: Segmenting customers into different profile groups for targeted marketing or grouping geographical areas by crime rates.
```
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0)
```
---
## Model Fitting
### Supervised Learning
```
lr.fit(X, y)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
```
### Unsupervised Learning
```
k_means.fit(X_train)
pca_model = pca.fit_transform(X_train)
```

| Type	       | Syntax Structure	| What it Means                                                                                                                           |
| ------------ | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Supervised	 | model.fit(X, y)  |	Requires two arguments: X (the features/input data) and y (the labels/correct target answers). The model learns the direct mapping from X to y. |
| Unsupervised | model.fit(X)     |	Requires only one argument: X (the input data). There is no y because the model is left on its own to figure out the geometric structure or groupings inherent in X. |

### Special Case: PCA's .fit_transform()
For dimensionality reduction models like PCA, you often see pca.fit_transform(X). This is a shortcut that does two things at once:
1. _fit(X)_: Learns the optimal structural angles (principal components) of the data.
2. _transform(X)_: Immediately applies those angles to compress the original dataset into a smaller footprint.
---
## Prediction
### Supervised Estimators
```
y_pred = svc.predict(np.random.random((2, 5)))
y_pred = lr.predict(X_test)
y_pred = knn.predict_proba(X_test)
```
### Unsupervised Estimators
```
y_pred = k_means.predict(X_test)
```

|Type	        | Method Used	          | What the Output (y_pred) Represents                                                                                                 |
| ----------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
|Supervised	  | predict(X_test)     	| Predicts an actual target value, like a specific category (e.g., "Spam") or a continuous number (e.g., "$350,000"). |
|Supervised  	| predict_proba(X_test) |	Output is a list of probabilities (e.g., "There is an 85% chance this email is spam, and a 15% chance it is not"). |
|Unsupervised	| predict(X_test)	| Assigns the new data to a specific cluster ID (e.g., Cluster 0, Cluster 1, Cluster 2) based on which trained center point it lives closest to. |

---
## Evaluate Your Model's Performance
### Classification Metrics
#### Accuracy Score
* **What it is**: It measures the percentage of correct predictions out of all total predictions made.
* **How it works**: It divides the number of correct guesses by the total number of test cases.
```
knn.score(X_test, y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```
> While easy to understand, it can be highly misleading if your dataset is imbalanced (e.g., if 99% of your emails are normal and 1% are spam, a broken model that guesses "not spam" every time will still get a 99% accuracy score).

#### Classification Report
* **What it is**: A comprehensive text summary that breaks down performance across multiple core metrics for each individual class.
* **How it works**: Instead of giving you one overall score, it splits the performance into a table containing three vital metrics for every category
| Metric    |	What It Answers	                                                                                                                 |   
| --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Precision |	Out of all the items the model claimed belonged to a class, how many were actually right? (Minimizes false alarms).	             |
| Recall    |	Out of all the actual real-world cases of a class, how many did the model successfully catch? (Minimizes missed targets).	       |
| F1-Score  |	The harmonic balance between Precision and Recall. It gives you a single, fair score when you need a compromise between the two. |

```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

#### Confusion Matrix
* **What it is**: A visual \(N \times N\) grid layout that maps out exactly where your model is getting confused by comparing Actual Labels against Predicted Labels.
* **How it works**: For a simple binary setup (like checking for a disease), it sorts every prediction into a four-quadrant table:
|	            | Predicted: NO                                                | Predicted: YES                                             |
| ----------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Actual: NO  | True Negative (TN) _(Correctly called healthy)_              | False Positive (FP) (False Alarm: Healthy but called sick) |
| Actual: YES	| False Negative (FN) (Missed Target: Sick but called healthy) | True Positive (TP) (Correctly called sick)                 |

```
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
```
