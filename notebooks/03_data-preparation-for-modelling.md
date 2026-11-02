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
