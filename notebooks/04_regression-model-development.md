# Build the Regression Model
## 4.1 Build a Linear Regression Model
Linear Regression is a supervised machine learning algorithm that predicts a numerical value by learning the relationship between input features and a target variable.

In this project, the model will learn how customer characteristics such as age, BMI, smoking status and region relate to insurance charges.

```
# Step 1: Import Linear Regression
from sklearn.linear_model import LinearRegression

# Step 2: Initialize the model
linear_model = LinearRegression()

# Step 3: Train the model
linear_model.fit(X_train_processed, y_train)

# Step 4: Generate predictions
y_pred = linear_model.predict(X_test_processed)

print("First 10 predicted charges:")
print(y_pred[:10])
```

Output:
```
First 10 predicted charges:
[ 8708.85457137  3887.67908457  7699.9894389  30058.27161708
  4452.39409091  5087.48833958 13051.45350769   493.0277202
 36062.49053772 10446.25283388]
```

### Interpretation:
The trained Linear Regression model was used to predict insurance charges for the testing dataset. The model generated 234 predictions based on the 11 preprocessed features. The first 10 predictions were displayed to verify that the model could successfully produce numerical outputs. These predictions will be compared with the actual insurance charges to evaluate the model's performance.

**One observation**: 
The model predicted RM493.03 for one record. This is below the project's minimum charge requirement of RM1,000. We will address this when generating the final validation predictions.

#### Step 3: Train the model
Understanding the code:
- `X_train_processed`: The 933 preprocessed training records containing 11 numerical features.
- `y_train`: The actual insurance charges for those 933 records.
- `.fit()`: Trains the model by learning the relationship between the input features and insurance charges.

#### Step 4: Generate predictions
Understanding the code:
- `linear_model.predict()`: Uses the trained model to generate predictions.
- `X_test_processed`: Contains the 234 testing records.
- `y_pred`: Stores the predicted insurance charges.
We will later compare these predictions against the actual charges in y_test to evaluate the model's performance.

## 4.2 Evaluate the model performance
Now, we need to evaluate how accurately our Linear Regression model predicts insurance charges.

We will use three evaluation metrics:
1. **R-squared (R²)**: Measures how much variation in insurance charges is explained by the model.
2. **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted charges.
3. **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared prediction errors, giving more weight to larger errors.

### 4.2.1 Calculate R-squared (R²)
R-squared is particularly important because the project requires an R² score greater than 0.65.
```
# Step 1: Import the evaluation metrics
from sklearn.metrics import r2_score as calculate_r2

# Step 2: Calculate the R² score
r2_value = calculate_r2(y_test, y_pred)
print("R-squared:", r2_value)
```

Output:
```
R-squared:  0.7621317165889079
```

Understanding the code:
* `y_test`: Actual insurance charges from the testing dataset.
* `y_pred`: Insurance charges predicted by the Linear Regression model.
* `r2_score()`: Compares actual and predicted values to measure how well the model explains the variation in insurance charges.

#### How to interpret R²?
| R² value | Interpretation                              |
|----------|---------------------------------------------|
| 1.0      | Perfect predictions                         |
| 0.8      | Strong explanatory performance              |
| 0.65     | Project's minimum target                    |
| 0.0      | No improvement over predicting the mean     |
| Below 0  | Worse than predicting the mean              |

> **Note**: R² is not a percentage of predictions that are exactly correct.

### Interpretation
The Linear Regression model achieved an R² score of 0.7621 on the testing dataset. This indicates that approximately 76.21% of the variation in insurance charges can be explained by the features used in the model. Since the score exceeds the project's minimum requirement of 0.65, the model meets the required R² performance target.

### 4.2.2 Calculate Mean Absolute Error (MAE)
Next, we will calculate the Mean Absolute Error to understand how far the model's predictions are from the actual insurance charges on average.

**Understanding MAE**:
MAE measures the average absolute difference between actual and predicted insurance charges.
For example, an MAE of RM3,000 means that the model's predictions differ from the actual charges by approximately RM3,000 on average.
```
# Step 1: Import the evaluation metrics
from sklearn.metrics import mean_absolute_error

# Step 2: Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
```
Output: 
```
Mean Absolute Error: 4037.321765254985
```

### Interpretation
The Linear Regression model achieved a Mean Absolute Error (MAE) of approximately RM4,037.32. This means that the model's predictions differ from the actual insurance charges by an average absolute amount of approximately RM4,037.32. A lower MAE indicates better prediction accuracy.

### 4.2.3 Calculate Root Mean Squared Error (RMSE)
Next, we will calculate RMSE to measure the prediction errors while giving greater weight to larger errors.

**Understanding RMSE**:
RMSE calculates the square root of the average squared differences between actual and predicted charges.
Unlike MAE, RMSE penalizes larger prediction errors more heavily. This helps us identify whether the model makes some predictions that are significantly different from the actual charges.

```
# Step 1: Import the evaluation metrics
from sklearn.metrics import mean_squared_error

# Step 2: Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)
```
Output: 
```
Root Mean Squared Error: 5693.123652143983
```

### Interpretation
The Linear Regression model achieved a Root Mean Squared Error (RMSE) of approximately RM5,693.12. This indicates that the model has a relatively larger prediction error when greater weight is given to larger differences between actual and predicted insurance charges. Since RMSE is higher than MAE, it suggests that some predictions have larger errors than others.

### 4.2.4 Summarize Model Performance
```
# Step 1: Create a performance summary
model_performance = pd.DataFrame({
    'Metric': ['R-squared', 'MAE', 'RMSE'],
    'Score': [r2_value, mae, rmse]
})

model_performance
```
Output:
| index | Metric    | Score          |
|-------|-----------|----------------|
| 0     | R-squared | 0.7621317166   |
| 1     | MAE       | 4037.321765255 |
| 2     | RMSE      | 5693.123652144 |

### Interpretation
The Linear Regression model achieved an R² score of 0.7621, indicating that it explains approximately 76.21% of the variation in insurance charges. The model recorded an MAE of RM4,037.32 and an RMSE of RM5,693.12. The higher RMSE compared to MAE suggests that some predictions have relatively larger errors. Overall, the model meets the project's minimum R² requirement of 0.65 and provides a baseline for comparison with other regression models.

## 4.3 Evaluate Model Generalization Using Cross-Validation
Before moving to more advanced models, let's check whether the Linear Regression model performs consistently across different subsets of the training data.

We will use **5-Fold Cross-Validation**.

**What is 5-Fold Cross-Validation?**
The training dataset is divided into five equal parts (folds).
* Four folds are used for training.
* One fold is used for validation.
* This process repeats five times, with each fold serving as the validation set once.

This helps us understand whether the model's performance is consistent rather than depending on just one train-test split.

### 4.3.1 Calculate cross-validation scores
```
# Step 1: Import the required function
from sklearn.model_selection import cross_val_score

# Step 2: Calculate cross-validation scores
cv_scores = cross_val_score(
    linear_model,
    X_train_processed,
    y_train,
    cv=5,
    scoring='r2'
)

# Step 3: Calculate the average cross-validation score
cv_mean = cv_scores.mean()

# Step 4: Calculate the standard deviation
cv_std = cv_scores.std()

print("Cross-validation R² scores:", cv_scores)
print("Average Cross-validation R²:", cv_mean)
print("Cross-validation R² standard deviation:", cv_std)
```

**Understanding the parameters**:
| Parameter                | Purpose                                            |
|--------------------------|----------------------------------------------------|
| `linear_model`           | The Linear Regression model being evaluated       |
| `X_train_processed`      | Preprocessed training features                     |
| `y_train`                | Actual training insurance charges                  |
| `cv=5`                   | Divides the training data into five folds          |
| `scoring='r2'`           | Uses R-squared to evaluate each fold               |

**Why calculate the average?**
The average R² score provides a more representative view of the model's performance across different training and validation subsets, rather than relying on a single fold.

**Why calculate standard deviation?**
* Lower standard deviation means the model performs more consistently across folds.
* Higher standard deviation means its performance varies more between folds.

> **Important note**: This evaluates the already-preprocessed training data. For a more rigorous cross-validation workflow, preprocessing should eventually be included inside the model pipeline so that imputation and scaling are refitted within each fold. We can improve this when comparing models.

Output:
```
Cross-validation R² scores: [0.71989275 0.70993113 0.64163667 0.74801304 0.70628518]
Average Cross-validation R²: 0.7051517569537633
Cross-validation R² standard deviation: 0.03496478056776199
```

### Interpretation
The Linear Regression model achieved an average cross-validation R² score of 0.7052 with a standard deviation of 0.0350. The relatively small standard deviation indicates that the model's performance does not vary significantly across the five folds. Although Fold 3 recorded a lower R² score of 0.6416, the remaining folds achieved scores above 0.70. Overall, the model demonstrates reasonably stable performance across different subsets of the training data.

### 4.3.2 Compare Testing and Cross-Validation Performance
| Evaluation              | R² Score |
|-------------------------|----------|
| Testing dataset         | 0.7621   |
| Average cross-validation | 0.7052   |
| Difference              | 0.0570   |

The testing score is approximately 0.0570 higher than the cross-validation average. This difference suggests some variation in performance depending on the data split, but it does not by itself establish overfitting.

**One important technical improvement before we continue**: 
Our earlier cross-validation used data that had already been preprocessed using the full training set. This means preprocessing information was shared across folds. For a more reliable cross-validation result, we should eventually put the preprocessor and Linear Regression model into one Pipeline and repeat the evaluation.
