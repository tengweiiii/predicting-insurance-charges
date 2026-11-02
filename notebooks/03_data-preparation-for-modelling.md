# Prepare Data for Modelling
Before splitting the dataset or building the regression model, we need to understand the relationship between the input features and the target variable, charges.
## 3.1 Separate Features and Target Variable
In supervised machine learning, the dataset is divided into two parts:
* **Features (X)**: Input variables used by the model to make predictions.
* **Target (y)**: The output variable that the model is trained to predict.
For this project:

| Type        | Columns                                      |
|-------------|----------------------------------------------|
| Features (X) | age, sex, bmi, children, smoker, region      |
| Target (y)   | charges                                      |

#### Step 1: Separate features and target
```
X = insurance_clean.drop(columns=['charges'])
y = insurance_clean['charges']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())
```
Output:
```
Features shape: (1167, 6)
Target shape: (1167,)

Feature columns:
['age', 'sex', 'bmi', 'children', 'smoker', 'region']
```
* X contains 1,167 records and 6 input features.
* y contains 1,167 target values.
* The `charges` column has been successfully separated from the input features.
> Why is this important? \nThe model learns from the six customer-related features in X and uses the actual insurance charges in y to learn how to make predictions.

## 3.2 Split Training and Testing Data
Before training the model, we need to split the dataset into training and testing sets.

**Why do we need this?**
* **Training data**: Used to teach the model patterns between customer information and insurance charges.
* **Testing data**: Used to evaluate how well the model predicts charges for data it has not seen before.

We will use an 80:20 split:
| Dataset  | Percentage | Purpose                      |
|----------|------------|------------------------------|
| Training | 80%        | Train the regression model   |
| Testing  | 20%        | Evaluate model performance   |

```
# Step 1: import train_test_split
from sklearn.model_selection import train_test_split

# Step 2: split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
```
* `test_size=0.2`: Reserves 20% of the data for testing.
* `random_state=42`: Ensures the same split is produced whenever you run the code.

Output:
```
Training features: (933, 6)
Testing features: (234, 6)
Training target: (933,)
Testing target: (234,)
```

### Interpretation
The cleaned dataset was divided into training and testing sets using an 80:20 ratio. The training set contains 933 records, while the testing set contains 234 records. The training data will be used to build the regression model, while the testing data will be used to evaluate its predictive performance.

## 3.3 Handle Missing Values and Encode Categorical Features
Before training the model, we need to address two issues:
1. **Missing values**: Numerical and categorical features still contain missing values.
2. **Categorical features**: Columns such as sex, smoker, and region contain text, which most regression algorithms cannot directly process.
   
We will use:
* **Median** imputation for numerical features.
* **Most frequent value** imputation for categorical features.
* **One-hot encoding** to convert categorical values into numerical columns.
We will build these transformations using Scikit-learn pipelines to ensure they are fitted on training data only, preventing data leakage.

### Step 1: Check missing values in training and testing sets
```
print("Training set missing values:")
print(X_train.isnull().sum())

print("\nTesting set missing values:")
print(X_test.isnull().sum())
```
Output:
```
Training set missing values:
age         29
sex         18
bmi         27
children    22
smoker      21
region      25
dtype: int64

Testing set missing values:
age          8
sex         11
bmi         10
children     7
smoker      10
region       6
dtype: int64
```

### 3.3.1 Review Missing Values
| Feature  | Data type   | Training missing | Testing missing | Treatment       |
|----------|-------------|------------------|-----------------|-----------------|
| age      | Numerical   | 29               | 8               | Median          |
| sex      | Categorical | 18               | 11              | Most frequent   |
| bmi      | Numerical   | 27               | 10              | Median          |
| children | Numerical   | 22               | 7               | Median          |
| smoker   | Categorical | 21               | 10              | Most frequent   |
| region   | Categorical | 25               | 6               | Most frequent   |

**Why do we use these methods?**
* **Median imputation**: Less affected by extreme values than the mean.
* **Most frequent imputation**: Replaces missing categorical values with the most common category.
> We will calculate the replacement values using the training set only. The testing set must not influence the imputation process.

### 3.3.2 Create Preprocessing Pipelines
We will use Scikit-learn's `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, and `ColumnTransformer`.
```
# Step 1: Import required libraries
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 2: Create Preprocessing Pipelines
numeric_features = ['age', 'bmi', 'children']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Step 3: Create the categorical preprocessing pipeline
categorical_features = ['sex', 'smoker', 'region']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Step 4: Combine both preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```
#### Step 1: Import required libraries
What does each component do?
| Component          | Purpose                                                   |
|--------------------|-----------------------------------------------------------|
| `SimpleImputer`    | Handles missing values                                   |
| `StandardScaler`   | Standardizes numerical features                          |
| `OneHotEncoder`    | Converts categorical features into numerical columns    |
| `ColumnTransformer` | Applies different preprocessing to different columns    |
| `Pipeline`         | Combines preprocessing steps into a workflow             |

#### Step 2: Create the numerical preprocessing pipeline
Our numerical features are:
* `age`
* `bmi`
* `children`
  
We will perform two preprocessing steps:
1. Replace missing values using the **median**.
2. Apply **StandardScaler** to standardize the numerical values.

**Understanding the code**:
| Step | Operation                         | Purpose                                                     |
|------|-----------------------------------|-------------------------------------------------------------|
| 1    | `SimpleImputer(strategy='median')` | Replaces missing numerical values with the median           |
| 2    | `StandardScaler()`                | Standardizes numerical values to a similar scale            |

For example, `age` and `BMI` have different measurement scales. Standardization helps prevent differences in scale from affecting models that are sensitive to feature magnitude.
> We are only defining the pipeline here. The median and scaling parameters will be learned later when we fit the pipeline on X_train.

#### Step 3: Create the categorical preprocessing pipeline
Our categorical features are:
* `sex`
* `smoker`
* `region`
  
We will perform two preprocessing steps:
1. Replace missing values using the most frequent value (**mode**).
2. Apply **One-Hot Encoding** to convert categorical values into numerical columns.

**Understanding the code**:
| Step | Operation                                  | Purpose                                                   |
|------|--------------------------------------------|-----------------------------------------------------------|
| 1    | `SimpleImputer(strategy='most_frequent')`  | Replaces missing values with the most common category    |
| 2    | `OneHotEncoder(handle_unknown='ignore')`   | Converts categories into numerical binary columns       |

> **Why use handle_unknown='ignore'?** \nThe testing dataset might contain a category that was not present in the training dataset. This setting prevents the encoder from raising an error when it encounters an unfamiliar category.

#### Step 4: Combine both preprocessing pipelines
We will use `ColumnTransformer` to apply the appropriate preprocessing steps to each group of features.

**Understanding the code**:
| Transformer | Features             | Operations                         |
|-------------|----------------------|------------------------------------|
| `num`       | age, bmi, children   | Median imputation + StandardScaler |
| `cat`       | sex, smoker, region  | Mode imputation + OneHotEncoder    |

The `ColumnTransformer` combines the processed numerical and categorical features into one dataset that can be used by a regression model.

> We are defining the preprocessing structure, not fitting it yet. It will learn the imputation values, scaling parameters and encoding categories from the training data when we fit the complete model pipeline.
