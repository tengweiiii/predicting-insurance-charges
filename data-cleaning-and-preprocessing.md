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

