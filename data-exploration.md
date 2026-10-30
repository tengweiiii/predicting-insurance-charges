# Data Exploration
## Data loading and exploration
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

### 2.1 Investigate missing values
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

### 2.2 Investigate invalid numerical values
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
  
> **Suggestion**: Treat negative ages and children as invalid records rather than converting them into positive values. For example, changing -40 into 40 assumes that the original value was entered with an incorrect sign. We don't have evidence to justify that assumption.

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

We should also preserve the original dataset:
```
insurance = insurance.copy()
```
This is useful when experimenting with different cleaning approaches.

### 2.4 Investigate missing value overlap
This is to understand whether missing values are concentrated in the same rows.
For example:
* If a row has all six features missing, removing it might be reasonable.
* If a row has only BMI missing but has valid charges, we might preserve it and impute BMI.
```
insurance[insurance.isnull().any(axis=1)]
```
Output: 
| index | age | sex    | bmi    | children | smoker | region    | charges      |
|-------|-----|--------|--------|----------|--------|-----------|--------------|
| 23    | -34 | female |        | 1        | yes    | null      | $37701.8768  |
| 32    |     | null   | 28.6   |          | null   | Southwest | $nan         |
| 43    | 37  | female |        |          | null   | southeast | 6313.759     |
| 44    |     | male   |        |          | no     | null      | null         |
| 49    |     | null   |        | 1        | null   | null      | null         |
| 51    | 21  | null   |        | 2        | null   | Northwest | $3579.8287   |
| 58    |     | null   | 22.88  |          | yes    | null      | 23244.7902   |
| 63    |     | female | 25.935 | 1        | null   | Northwest | 4133.64165   |
| 65    |     | null   | 28.9   |          | null   | Southwest | 1743.214     |
| 76    |     | F      |        | 1        | no     | Southeast | $nan         |
| 78    | 22  | F      | 39.805 |          | no     | Northeast | null         |
| 101   |     | male   |        |          | null   | northeast | 3645.0894    |
| 115   | 60  | male   |        | 0        | null   | Northeast | null         |
| 163   | 32  | null   | 29.8   | 2        | null   | null      | null         |
| 184   | -44 | null   | 30.69  |          | no     | null      | $7731.4271   |
| 208   | 63  | null   | 31.8   | 0        | no     | southwest | null         |
| 210   |     | null   | 33     | 1        | null   | null      | 1980.07      |
| 221   | 53  | null   | 33.25  | 0        | null   | Northeast | null         |
| 233   |     | null   |        |          | no     | Southwest | null         |
| 237   |     | null   |        |          | null   | Southeast | 4463.2051    |
| 240   |     | female | 36.67  |          | yes    | null      | null         |
| 247   |     | man    |        |          | no     | Southeast | null         |
| 254   | 50  | male   | 31.825 | 0        | null   | null      | 41097.16175  |
| 259   |     | male   |        | 0        | null   | null      | 33750.2918   |
| 265   | 46  | male   | 42.35  |          | null   | southeast | 46151.1245   |
| 270   |     | male   |        | 1        | no     | null      | $1719.4363   |
| 275   |     | female |        |          | no     | Northeast | 9715.841     |
| 289   |     | null   | 26.4   | 3        | no     | null      | null         |
| 298   | 31  | null   |        | 3        | yes    | Northwest | $nan         |
| 303   |     | female |        |          | no     | null      | 4349.462     |
| 309   |     | null   |        |          | no     | null      | 7749.1564    |
| 312   | 43  | null   |        | 3        | yes    | southeast | null         |
| 346   |     | male   |        |          | null   | southeast | null         |
| 358   | 23  | male   |        | 0        | no     | southeast | $nan         |
| 363   |     | female | 26.4   |          | no     | null      | null         |
| 365   | 49  | female | 30.78  | 1        | no     | null      | null         |
| 367   | 42  | null   |        | 2        | no     | Northwest | 8017.06115   |
| 374   |     | null   | 33.33  |          | no     | null      | 1391.5287    |
| 394   | 46  | female | 32.3   |          | no     | northeast | null         |
| 405   |     | female |        |          | null   | null      | 11396.9002   |
| 413   | 25  | null   | 23.9   |          | no     | southwest | null         |
| 415   | 43  | F      |        | 1        | no     | null      | $7345.7266   |
| 429   |     | female |        |          | no     | Northwest | $nan         |
| 430   | 19  | male   |        |          | null   | Southwest | 23082.95533  |
| 447   |     | null   |        |          | null   | null      | null         |
| 462   | 62  | null   |        |          | no     | northeast | 15230.32405  |
| 494   |     | null   | 25.7   |          | null   | null      | null         |
| 506   | 22  | male   | 31.35  | 1        | null   | null      | null         |
| 513   |     | male   |        | 0        | no     | null      | 1256.299     |
| 538   |     | null   | 28.05  | -1       | no     | Southeast | null         |
| 542   | 63  | female | 36.3   | 0        | null   | null      | 13887.204    |
| 551   | 32  | woman  | 28.93  |          | null   | southeast | 3972.9247    |
| 561   |     | null   |        |          | no     | null      | null         |
| 567   |     | null   | 30.59  | 2        | no     | northwest | 7256.7231    |
| 572   | 30  | female |        | 2        | no     | null      | null         |
| 594   | 41  | male   |        |          | no     | null      | $5709.1644   |
| 597   | 34  | female | 33.25  | 1        | null   | null      | 5594.8455    |
| 598   | 43  | null   | 32.6   | 2        | null   | null      | null         |
| 614   |     | F      |        | 0        | null   | null      | $1880.07     |
| 615   | -47 | null   | 36.63  |          | null   | southeast | $nan         |
| 650   |     | null   | 42.68  | 2        | no     | Southeast | $9800.8882   |
| 654   |     | null   | 35.2   | 0        | null   | null      | null         |
| 664   | 64  | F      |        | 0        | null   | null      | $nan         |
| 678   | 56  | null   |        | 3        | null   | null      | null         |
| 700   | 21  | null   | 34.87  |          | null   | Southeast | $nan         |
| 707   |     | null   |        |          | no     | Northwest | 10264.4421   |
| 710   | 18  | null   |        |          | null   | null      | null         |
| 712   | 43  | female | 30.685 |          | no     | null      | null         |
| 721   | 53  | male   | 36.6   | 3        | no     | southwest | null         |
| 752   | 64  | null   | 37.905 | 0        | no     | Northwest | 14210.53595  |
| 755   |     | male   |        | 2        | no     | null      | 5031.26955   |
| 764   |     | female | 25.175 | 2        | null   | Northeast | null         |
| 765   |     | female | 31.825 | 0        | null   | Northwest | null         |
| 770   |     | man    |        | 3        | null   | southwest | 27941.28758  |
| 774   | -41 | man    | 34.2   | 2        | null   | null      | 7261.741     |
| 777   |     | man    | 39.805 | 0        | no     | northeast | null         |
| 778   | 35  | M      | 34.32  |          | null   | null      | 5934.3798    |
| 780   |     | null   |        |          | null   | null      | null         |
| 793   | 53  | null   | 20.9   |          | null   | null      | 21195.818    |
| 798   | 58  | F      | 33.1   |          | no     | null      | null         |
| 806   | 40  | null   |        | 1        | null   | null      | 28476.73499  |
| 808   | 18  | male   | 30.14  | 0        | no     | null      | 1131.5066    |
| 818   | 47  | null   | 26.125 | -1       | yes    | null      | null         |
| 832   | 28  | female | 23.845 |          | null   | null      | null         |
| 838   | 35  | null   |        |          | no     | null      | $6402.29135  |
| 849   | -55 | null   | 32.775 |          | null   | null      | 10601.63225  |
| 868   | 61  | null   | 23.655 |          | no     | northeast | 13129.60345  |
| 869   |     | null   | 24.3   | 3        | null   | null      | $nan         |
| 881   | 23  | null   |        | 1        | null   | null      | null         |
| 885   |     | male   | 28.93  | 1        | null   | null      | null         |
| 887   | 36  | null   | 30.02  |          | null   | null      | null         |
| 888   |     | male   |        | 0        | null   | null      | 1682.597     |
| 890   | -64 | female | 26.885 |          | null   | Northwest | 29330.98315  |
| 893   |     | null   |        | 2        | yes    | southeast | 44202.6536   |
| 899   | 19  | null   | 22.515 |          | no     | null      | null         |
| 901   | 60  | M      |        | 0        | null   | southeast | null         |
| 904   |     | woman  | 35.1   |          | null   | southwest | null         |
| 906   |     | man    |        | 3        | null   | null      | 4846.92015   |
| 916   |     | null   | 26.885 | 0        | null   | Northwest | null         |
| 925   |     | null   |        |          | null   | Northeast | $25333.33284 |
| 930   | 26  | null   | 46.53  | 1        | null   | null      | 2927.0647    |
| 945   |     | female |        |          | null   | null      | 11674.13     |
| 952   |     | female |        | 1        | no     | northwest | 4527.18295   |
| 979   | 36  | null   |        |          | no     | Southeast | null         |
| 1015  | 59  | male   |        |          | null   | Northwest | $nan         |
| 1047  |     | null   |        | 1        | null   | southeast | 44501.3982   |
| 1048  |     | null   | 22.515 |          | no     | Northwest | 3594.17085   |
| 1052  |     | male   |        | 1        | null   | northeast | $9288.0267   |
| 1098  |     | null   |        | 0        | null   | null      | $nan         |
| 1108  |     | man    | 30     |          | no     | southwest | null         |
| 1118  |     | null   |        |          | yes    | null      | $38282.7495  |
| 1119  |     | null   |        | 3        | no     | null      | 5693.4305    |
| 1127  | 35  | null   |        |          | no     | southeast | null         |
| 1135  |     | null   | 28.12  | 3        | no     | northwest | 11085.5868   |
| 1143  |     | null   |        |          | no     | southeast | 6338.0756    |
| 1171  |     | female |        |          | yes    | Southwest | 22478.6      |
| 1176  | 52  | null   |        |          | null   | null      | null         |
| 1212  |     | null   | 21.47  | 0        | no     | null      | null         |
| 1228  | 56  | null   |        | 0        | no     | null      | 10594.2257   |
| 1246  | 45  | female | 25.7   |          | null   | southwest | null         |
| 1253  |     | null   |        |          | null   | Southwest | null         |
| 1263  | 43  | null   | 29.9   |          | null   | null      | 7337.748     |
| 1280  | 48  | null   |        |          | no     | Southeast | null         |
| 1286  | 28  | female | 17.29  | 0        | no     | null      | null         |
| 1293  | 46  | man    |        |          | null   | null      | 9301.89355   |
| 1296  |     | null   | 26.125 | 0        | null   | northeast | 1708.92575   |
| 1309  |     | male   |        |          | no     | Southwest | null         |
| 1312  |     | M      |        | 1        | null   | Southwest | 4536.259     |
| 1314  | 30  | null   | 23.655 |          | yes    | northwest | null         |
| 1321  |     | male   | 26.695 |          | yes    | null      | 28101.33305  |

```
insurance[insurance.isnull().all(axis=1)]
```
Output: 
| index | age  | sex  | bmi | children | smoker | region | charges |
|-------|------|------|-----|----------|--------|--------|---------|
| 447   | null |      |     |          | null   | null   | null    |
| 780   | null |      |     |          | null   | null   | null    |


### 2.5 Investigate the charges column
This is to understand the currency symbol and $nan values before converting the column.
```
insurance['charges'].head(20)
```
Output: 
| index | charges      |
|-------|--------------|
| 0     | 16884.924    |
| 1     | 1725.5523    |
| 2     | $4449.462    |
| 3     | $21984.47061 |
| 4     | $3866.8552   |
| 5     | $3756.6216   |
| 6     | 8240.5896    |
| 7     | 7281.5056    |
| 8     | $6406.4107   |
| 9     | 28923.13692  |
| 10    | $2721.3208   |
| 11    | 27808.7251   |
| 12    | 1826.843     |
| 13    | 11090.7178   |
| 14    | 39611.7577   |
| 15    | 1837.237     |
| 16    | 10797.3362   |
| 17    | $2395.17155  |
| 18    | 10602.385    |
| 19    | 36837.467    |

```
insurance['charges'].unique()[:20]
```
Output: 
```
array(['16884.924', '1725.5523', '$4449.462', '$21984.47061',
       '$3866.8552', '$3756.6216', '8240.5896', '7281.5056', '$6406.4107',
       '28923.13692', '$2721.3208', '27808.7251', '1826.843',
       '11090.7178', '39611.7577', '1837.237', '10797.3362',
       '$2395.17155', '10602.385', '36837.467'], dtype=object)
```

### Interpretation of the Findings

There are only 2 rows have every column missing, at indices 447 and 780. Most other rows with missing values still contain useful information. 

| Situation | Example | Possible treatment |
|---|---|---|
| Entire row is empty | Index 447 and 780 | Remove the rows |
| Some features are missing, but charges are available | Age or BMI is missing | Impute missing features or remove selected rows |
| Charges are missing | `$nan` or `null` | Convert these to missing values and exclude these rows from model training |

We cannot train a supervised regression model using rows without a known target (charges). The model needs actual charges to learn the relationship between customer characteristics and insurance costs.

However, we should not automatically remove every row with missing features. That could discard useful training data.

Based on the first 20 records and the unique values, we can identify two main inconsistencies in the charges column:
1. **Inconsistent currency symbol**: Some values contain a $ symbol, while others contain only numeric characters. For example, 16884.924 and $4449.462 represent the same type of information but are stored in different formats.
2. **Object data type**: The column has the object data type because the values are stored as strings rather than numeric values. This prevents us from directly performing numerical calculations or using the column as a regression target.
3. **Potential missing values**: Previous investigations also found $nan values. These represent missing or invalid charge values and should be converted to actual NaN values.
> The charges column requires cleaning before modelling. We need to remove the $ symbol and convert the values into a numeric data type. Invalid or missing values should become NaN so that we can handle them properly.
