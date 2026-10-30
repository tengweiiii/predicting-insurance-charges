# Data Cleaning and Preprocessing
## Step 1: Data loading and exploration
```
# Import required libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Loading the insurance dataset
insurance_data_path = 'insurance.csv'
insurance = pd.read_csv(insurance_data_path)

insurance.head()
```

### 1.1 Understand the dataset
#### Task A: Check the dataset dimensions
```
insurance.shape
```
Output: 
```
(1338, 7)
```
Observation:
Total rows: 1338 (Dataset size)
Total columns: 7 (6 features + 1 target)

#### Task B: Check the column information
```
insurance.info()
```
Pay attention to:
* Number of entries.
* Data types (int64, float64, object).
* Non-null counts.
* Whether there are missing values.
  
Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1272 non-null   float64
 1   sex       1272 non-null   object 
 2   bmi       1272 non-null   float64
 3   children  1272 non-null   float64
 4   smoker    1272 non-null   object 
 5   region    1272 non-null   object 
 6   charges   1284 non-null   object 
dtypes: float64(3), object(4)
memory usage: 73.3+ KB
```
Observation:
| Item     | Result | Observation |
|--------- | -------- | -------- |
| Age | 1,272 non-null	| 66 missing |
| Sex | 1,272 non-null	| 66 missing |
| BMI	| 1,272 non-null	| 66 missing |
| Children | 1,272 non-null	| 66 missing |
| Smoker| 1,272 non-null	| 66 missing |
| Region	| 1,272 non-null	| 66 missing |
| Charges	| 1,284 non-null	| 54 missing |

> Notice that the first six columns have exactly the same number of non-null values. This might indicate that some records have multiple missing fields.

#### Task C: Generate descriptive statistics
```
insurance.describe()
```
This gives us:
* Mean and standard deviation.
* Minimum and maximum values.
* 25th, 50th and 75th percentiles.
  
Output: 
| index	| age	| bmi	| children |
| -------- | --------- | ---------- | ---------- |
| count	| 1272	| 1272	| 1272 |
| mean	| 35.2146226415	| 30.5605503145	| 0.9488993711 |
| std	| 22.4782508402	| 6.0955733842	| 1.3035322282 |
| min	| -64	| 15.96	| -4 |
| 25%	| 24.75	| 26.18	| 0 |
| 50%	| 38	| 30.21	| 1 |
| 75%	| 51	| 34.485	| 2 |
| max	| 64	| 53.13 |	5 |

#### Task D: Check categorical values
```
insurance.describe(include='object')
```
Output: 
| index	| sex	| smoker	| region	| charges |
| -------- | -------- | -------- | -------- | -------- |
| count	| 1272	| 1272	| 1272	| 1284 |
| unique	| 6	| 2	| 8	| 1272 |
| top	| male	| no	| Southeast	| $nan |
| freq	| 517	| 1013	| 172	| 12 |

### Issues discovered
#### Issue 1: Missing values - Needs investigation
There are 66 missing values in each of the 6 feature columns and 54 missing values in charges.

#### Issue 2: Invalid age - Anomaly
The minimum age is -64. A person's age cannot be negative

#### Issue 3: Invalid children - Anomaly
The minimum children value is -4. The number of dependents cannot be negative.

#### Issue 4: Categorical inconsistencies - Needs investigation
* sex has 6 unique values, although only male and female are expected.
* region has 8 unique values, although only 4 regions are expected.

#### Issue 5: Incorrect data type - Needs conversion
`charges` is currently an object instead of a numeric column.

Your output also shows `$nan` as the most frequent value, suggesting that some charges may contain currency symbols or text representations of missing values.

## Data Cleaning and Preprocessing
### Task 1: Investigate missing values
```
insurance.isnull().sum()
```
Output: 
| index	| 0 |
| ----- | ----- |
| age	| 66 |
| sex	| 66 |
| bmi	| 66 |
| children	| 66 |
| smoker	| 66 |
| region	| 66 |
| charges	| 54 |

### Task 1.1: Calculate the missing percentage
```
(insurance.isnull().sum() / len(insurance)) * 100
```
Output:
| index	| 0 |
| ----- | --------------- |
| age	| 4.932735426 |
| sex	| 4.932735426 |
| bmi	| 4.932735426 |
| children	| 4.932735426 |
| smoker	| 4.932735426 |
| region	| 4.932735426 |
| charges	| 4.0358744395 |
