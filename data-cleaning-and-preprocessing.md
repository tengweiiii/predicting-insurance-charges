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
#### 1.1.1 Check the dataset dimensions
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

#### 1.1.2 Check the column information
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

#### 1.1.3: Generate descriptive statistics
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

#### 1.1.4: Check categorical values
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

## Step 2 Data Cleaning and Preprocessing
### 2.1: Investigate missing values
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

Then, calculate the missing percentage: 
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

### 2.2: Investigate invalid numerical values
#### Age
```
insurance[insurance['age'] < 0]
```
Output: 
| index | age | sex    | bmi    | children | smoker | region    | charges      |
|-------|-----|--------|--------|----------|--------|-----------|--------------|
| 5     | -31 | woman  | 25.74  | 0        | no     | southeast | $3756.6216   |
| 14    | -27 | male   | 42.13  | 0        | yes    | southeast | 39611.7577   |
| 17    | -23 | male   | 23.845 | 0        | no     | northeast | $2395.17155  |
| 23    | -34 | female |        | 1        | yes    | null      | $37701.8768  |
| 79    | -41 | female | 32.965 | 0        | no     | Northwest | $6571.02435  |
| 93    | -35 | male   | 34.77  | 2        | no     | Northwest | $5729.0053   |
| 179   | -41 | female | 33.155 | 3        | no     | Northeast | $8538.28845  |
| 184   | -44 | null   | 30.69  |          | no     | null      | $7731.4271   |
| 196   | -39 | female | 32.8   | 0        | no     | Southwest | $5649.715    |
| 205   | -28 | female | 28.88  | 1        | no     | Northeast | $4337.7352   |
| 211   | -40 | male   | 30.875 | 4        | no     | Northwest | $8162.71625  |
| 292   | -25 | M      | 45.54  | 2        | yes    | southeast | $42112.2356  |
| 315   | -52 | male   | 33.25  | 0        | no     | northeast | $9722.7695   |
| 321   | -26 | female | 29.64  | 4        | no     | Northeast | $24671.66334 |
| 384   | -44 | male   | 22.135 | 2        | no     | Northeast | 8302.53565   |
| 401   | -47 | male   | 47.52  | 1        | no     | southeast | 8083.9198    |
| 445   | -45 | female | 33.1   | 0        | no     | Southwest | 7345.084     |
| 457   | -57 | female | 30.495 | 0        | no     | Northwest | $11840.77505 |
| 497   | -45 | male   | 28.7   | 2        | no     | southwest | 8027.968     |
| 501   | -43 | male   | 26.03  | 0        | no     | northeast | 6837.3687    |
| 563   | -50 | male   | 44.77  | 1        | no     | southeast | $9058.7303   |
| 590   | -58 | female | 29     | 0        | no     | Southwest | $11842.442   |
| 599   | -52 | woman  | 37.525 | 2        | no     | Northwest | 33471.97189  |
| 615   | -47 | null   | 36.63  |          | null   | southeast | $nan         |
| 616   | -56 | female | 28.595 | 0        | no     | northeast | 11658.11505  |
| 620   | -30 | male   | 31.4   | 1        | no     | Southwest | $3659.346    |
| 634   | -51 | male   | 39.7   | 1        | no     | southwest | $9391.346    |
| 655   | -52 | female | 25.3   | 2        | yes    | Southeast | 24667.419    |
| 670   | -30 | male   | 31.57  | 3        | no     | southeast | $nan         |
| 688   | -47 | female | 24.1   | 1        | no     | southwest | $26236.57997 |
| 691   | -47 | male   | 36.2   | -1       | no     | southwest | $8068.185    |
| 715   | -60 | man    | 28.9   | 0        | no     | Southwest | $12146.971   |
| 720   | -51 | female | 40.66  | 0        | no     | northeast | 9875.6804    |
| 750   | -37 | F      | 26.4   | 0        | yes    | southeast | 19539.243    |
| 757   | -47 | female | 27.83  | 0        | yes    | southeast | 23065.4207   |
| 774   | -41 | man    | 34.2   | 2        | null   | null      | 7261.741     |
| 820   | -45 | man    | 33.7   | 1        | no     | southwest | 7445.918     |
| 828   | -41 | M      | 30.78  | 3        | yes    | Northeast | 39597.4072   |
| 849   | -55 | null   | 32.775 |          | null   | null      | 10601.63225  |
| 850   | -37 | female | 30.78  | 0        | yes    | northeast | 37270.1512   |
| 864   | -51 | male   | 25.4   | 0        | no     | southwest | 8782.469     |
| 879   | -37 | female | 29.5   | 2        | no     | Southwest | 6311.952     |
| 890   | -64 | female | 26.885 |          | null   | Northwest | 29330.98315  |
| 927   | -57 | female | 20.1   | 1        | no     | Southwest | $12032.326   |
| 937   | -39 | female | 24.225 | 5        | no     | Northwest | $8965.79575  |
| 972   | -22 | woman  | 20.235 | 0        | no     | Northwest | 2527.81865   |
| 999   | -36 | woman  | 26.885 | 0        | no     | northwest | 5267.81815  |
| 1025  | -21 | female | 34.6   | 0        | no     | Southwest | 2020.177     |
| 1041  | -18 | male   | 23.085 | 0        | no     | Northeast | 1704.70015   |
| 1055  | -55 | male   | 27.645 | 0        | no     | Northwest | 10594.50155  |
| 1063  | -36 | male   | 28.595 | 3        | no     | northwest | $6548.19505  |
| 1142  | -52 | female | 24.86  | 0        | no     | Southeast | 27117.99378  |
| 1144  | -50 | male   | 32.3   | 2        | no     | southwest | 9630.397     |
| 1147  | -20 | F      | 31.92  | 0        | no     | northwest | 2261.5688    |
| 1150  | -18 | female | 30.305 | 0        | no     | northeast | 2203.73595  |
| 1216  | -40 | man    | 25.08  | 0        | no     | Southeast | 5415.6612    |
| 1251  | -19 | male   | 19.8   | 0        | no     | southwest | 1241.565     |
| 1254  | -34 | F      | 27.72  | 0        | no     | southeast | $4415.1588   |
| 1265  | -64 | male   | 23.76  | 0        | yes    | Southeast | $26926.5144  |
| 1273  | -35 | M      | 27.61  | 1        | no     | southeast | 4747.0529    |
| 1278  | -39 | male   | 29.925 | 1        | yes    | northeast | 22462.04375  |
| 1291  | -19 | male   | 34.9   | 0        | yes    | southwest | 34828.654    |
| 1303  | -43 | male   | 27.8   | 0        | yes    | southwest | 37829.7242   |
| 1311  | -33 | female | 26.695 | 0        | no     | northwest | 4571.41305   |
| 1334  | -18 | female | 31.92  | 0        | no     | Northeast | 2205.9808    |

```
insurance[insurance['children'] < 0]
```
Output: 
| index | age | sex    | bmi    | children | smoker | region    | charges     |
|-------|-----|--------|--------|----------|--------|-----------|-------------|
| 34    | 28  | male   | 36.4   | -1       | yes    | Southwest | 51194.55914 |
| 66    | 61  | F      | 39.1   | -2       | no     | southwest | 14235.072   |
| 84    | 37  | female | 34.8   | -2       | yes    | Southwest | $39836.519  |
| 105   | 20  | male   | 28.025 | -1       | yes    | Northwest | 17560.37975 |
| 129   | 38  | male   | 34.7   | -2       | no     | southwest | 6082.405    |
| 185   | 36  | male   | 41.895 | -3       | yes    | Northeast | 43753.33705 |
| 225   | 55  | male   | 33.88  | -3       | no     | southeast | 11987.1682  |
| 243   | 40  | male   | 35.3   | -3       | no     | southwest | 7196.867    |
| 267   | 59  | F      | 32.395 | -3       | no     | northeast | 14590.63205 |
| 279   | 51  | female | 21.56  | -1       | no     | Southeast | $9855.1314  |
| 297   | 47  | M      | 25.41  | -1       | yes    | Southeast | $21978.6769 |
| 398   | 64  | male   | 25.6   | -2       | no     | southwest | 14988.432   |
| 431   | 29  | female | 20.235 | -2       | no     | northwest | 4906.40965  |
| 538   |     | null   | 28.05  | -1       | no     | Southeast | null        |
| 571   | 18  | female | 37.29  | -1       | no     | Southeast | 2219.4451   |
| 578   | 52  | M      | 30.2   | -1       | no     | southwest | 9724.53     |
| 583   | 32  | F      | 23.65  | -1       | no     | southeast | 17626.23951 |
| 621   | 37  | man    | 34.1   | -4       | yes    | Southwest | 40182.246   |
| 645   | 48  | male   | 30.78  | -3       | no     | northeast | $10141.1362 |
| 653   | 45  | female | 36.3   | -2       | no     | southeast | $8527.532   |
| 662   | 32  | female | 31.54  | -1       | no     | Northeast | 5148.5526   |
| 667   | 40  | female | 32.775 | -2       | yes    | Northwest | 40003.33225 |
| 669   | 40  | female | 29.81  | -1       | no     | southeast | 6500.2359   |
| 691   | -47 | male   | 36.2   | -1       | no     | southwest | $8068.185   |
| 811   | 33  | female | 42.94  | -3       | no     | northwest | $6360.9936  |
| 818   | 47  | null   | 26.125 | -1       | yes    | null      | null        |
| 853   | 53  | female | 23.75  | -2       | no     | northeast | 11729.6795  |
| 938   | 18  | male   | 26.18  | -2       | no     | southeast | 2304.0022   |
| 941   | 50  | F      | 46.09  | -1       | no     | Southeast | 9549.5651   |
| 962   | 63  | female | 35.2   | -1       | no     | southeast | 14474.675   |
| 968   | 21  | male   | 25.745 | -2       | no     | northeast | 3279.86855  |
| 994   | 27  | female | 20.045 | -3       | yes    | Northwest | 16420.49455 |
| 1005  | 29  | male   | 31.73  | -2       | no     | northwest | 4433.3877   |
| 1009  | 51  | man    | 27.74  | -1       | no     | northeast | 9957.7216   |
| 1017  | 26  | female | 34.2   | -2       | no     | southwest | $3987.926   |
| 1021  | 22  | female | 31.02  | -3       | yes    | southeast | 35595.5898  |
| 1079  | 63  | male   | 33.66  | -3       | no     | Southeast | 15161.5344  |
| 1083  | 32  | male   | 31.5   | -1       | no     | Southwest | 4076.497    |
| 1094  | 50  | female | 33.7   | -4       | no     | southwest | $11299.343  |
| 1095  | 18  | woman  | 31.35  | -4       | no     | Northeast | $4561.1885  |
| 1128  | 34  | male   | 32.8   | -1       | no     | southwest | 14358.36437 |
| 1191  | 41  | woman  | 21.755 | -1       | no     | northeast | 13725.47184 |
| 1224  | 41  | male   | 23.94  | -1       | no     | Northeast | 6858.4796   |
| 1299  | 19  | female | 25.745 | -1       | no     | northwest | $2710.82855 |

#### BMI
```
insurance['bmi'].describe()
```
Output:
| index                | bmi   |
| -------------------- | ----- |
| Count                | 1,272 |
| Mean                 | 30.56 |
| Standard Deviation   | 6.10  |
| Minimum              | 15.96 |
| 25th Percentile (Q1) | 26.18 |
| Median (Q2)          | 30.21 |
| 75th Percentile (Q3) | 34.49 |
| Maximum              | 53.13 |

### 2.3 Investigate categorical inconsistencies
#### Sex
```
insurance['sex'].value_counts(dropna=False)
```
Output: 
| index    | sex |
| ------ | ----- |
| Male   | 517   |
| Female | 503   |
| Null   | 66    |
| Man    | 64    |
| M      | 64    |
| Woman  | 62    |

#### Smoker
```
insurance['smoker'].value_counts(dropna=False)
```
Output:
| Smoker | Count |
|--------|------:|
| No     | 1,013 |
| Yes    | 259   |
| Null   | 66    |

#### Region
```
insurance['region'].value_counts(dropna=False)
```
| index     | region |
|-----------|------:|
| Southeast | 172   |
| southeast | 170   |
| southwest | 164   |
| Northwest | 161   |
| Northeast | 155   |
| northeast | 153   |
| northwest | 149   |
| Southwest | 148   |
| Null      | 66    |

### Review: Data quality assessment
#### Missing values
All 6 feature columns have exactly 66 missing values. This suggestd that some rows might have been corrupted or partially removed.
> Suggestion: Do not immediate;y remove every row containing a first value. Investigate whether the missing values occur in the same records.

#### Invalid numerical values
* **Age**: Negative ages are impossible. These records need correction or removal
* **Children**: Negatove numbers of children are also impossible
* **BMI**: No obvious invalid BMI values for the descriptive statistics

One important distinction:
* Negative age is an invalid value.
* Negative children is an invalid value.
* High BMI is not automatically an error.
  
> Suggestion: Treat negative ages and children as invalid records rather than converting them into positive values. For example, changing -40 into 40 assumes that the original value was entered with an incorrect sign. We don't have evidence to justify that assumption.

#### Categorical inconsistencies
* **Sex**: standardize into lowercase male and female
* **Smoker**: standardize the capitalization
* **Region**: standardize the capitalization
> Convert categorical text to lowercase and remove unnecessary spaces before checking the unique values again.

### Recommended cleaning strategy
| Problem | Recommended treatment |
|---|---|
| Missing feature values | Investigate overlap, then consider imputation or row removal |
| Missing `charges` | Remove records without a target value from training |
| Negative `age` | Remove invalid records |
| Negative `children` | Remove invalid records |
| Sex inconsistencies | Standardize into `male` / `female` |
| Smoker capitalization | Convert to lowercase |
| Region capitalization | Convert to lowercase |
| Charges currency symbols | Remove symbols and convert to numeric |
