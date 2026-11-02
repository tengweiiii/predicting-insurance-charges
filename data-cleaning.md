# Data Cleaning

## 2.1 Overview

After completing the initial data exploration, several data quality issues were identified in the insurance dataset. These include missing values, inconsistent categorical values, invalid numerical values, and inconsistent formatting in the `charges` column.

The purpose of this stage is to clean and prepare the dataset so that it can be used for regression modelling.

## 2.2 Data Cleaning Objectives

The following issues will be addressed:

| Column | Identified Issue | Cleaning Approach |
|---|---|---|
| `age` | Missing and negative values | Handle missing and invalid values |
| `sex` | Inconsistent labels such as Male, Man and M | Standardize categories |
| `bmi` | Missing values | Handle missing values |
| `children` | Missing and negative values | Handle missing and invalid values |
| `smoker` | Missing values | Handle missing values |
| `region` | Inconsistent capitalization | Standardize text |
| `charges` | Currency symbols and missing values | Convert to numeric and remove records with missing targets |

## 2.3 Clean the Target Column: `charges`

The `charges` column is the target variable that the regression model will predict. Therefore, it must be converted into a numerical format before model training.

### 2.3.1 Remove Currency Symbols and Convert to Numeric

Some values contain a `$` symbol, while others are stored as numeric-looking strings. The column also contains invalid values such as `$nan`.

The values were converted into numeric format, with invalid entries treated as missing values.

```
insurance['charges'] = insurance['charges'].astype(str).str.replace('$', '', regex=False)

insurance['charges'] = pd.to_numeric(
    insurance['charges'],
    errors='coerce'
)

print("Data type:")
print(insurance['charges'].dtype)

print("\nMissing values:")
print(insurance['charges'].isnull().sum())

print("\nSummary statistics:")
print(insurance['charges'].describe())
```
* `.str.replace('$', '', regex=False)` removes the $ symbol without treating it as a regular expression.
* `pd.to_numeric()` converts the cleaned strings into numbers.
* `errors='coerce'` converts values that cannot be interpreted as numbers into NaN.

Output:
```
Data type:
float64

Missing values:
66

Summary statistics:
count     1272.000000
mean     13286.594477
std      12142.505233
min       1121.873900
25%       4733.582163
50%       9382.033000
75%      16579.959053
max      63770.428010
Name: charges, dtype: float64
```
#### Data type 
```
float64
```
**Interpretation**: Successfully converted `charges` from `object` to `float64`. The column can now be used for numerical calculations and regression modelling.
#### Missing values 
```
66
```
**Interpretation**: There are now 66 missing values in `charges`, compared with the original 54.
> **Why did the count increase?** \nSome values, such as $nan, were previously stored as strings. After conversion, pandas recognized them as missing values. This is expected. We should remove rows with missing target values before training the model.
#### Summary statistics
| Statistic | Value | Interpretation |
|---|---:|---|
| Count | 1,272 | Number of records with valid charges |
| Mean | 13,286.59 | Average insurance charge |
| Standard deviation | 12,142.51 | Charges vary considerably |
| Minimum | 1,121.87 | Lowest observed charge |
| 25th percentile | 4,733.58 | 25% of valid charges are at or below this value |
| Median | 9,382.03 | Half of valid charges are at or below this value |
| 75th percentile | 16,579.96 | 75% of valid charges are at or below this value |
| Maximum | 63,770.43 | Highest observed charge |

**Key finding**: The mean is higher than the median, suggesting that the distribution of insurance charges is right-skewed. Some individuals have substantially higher charges than others.

The minimum charge is positive, and the maximum is plausible for an insurance dataset. There are no obvious problems in these summary statistics, although we should investigate unusual values further if needed.

### 2.3.2 Remove Rows with Missing Charges
Since `charges` is our **target variable**, we need to remove records where the charge amount is missing. A regression model cannot learn from a training record without a known target value.
#### Step 1: Check the number of missing values
```
insurance_clean['charges'].isnull().sum()
```
Output:
```
66
```

#### Step 2: Remove rows with missing charges
Use `dropna()` with `subset=['charges']` so that only rows with missing target values are removed.
```
insurance_clean = insurance_clean.dropna(subset=['charges'])
print("Dataset shape:", insurance_clean.shape)
print("Missing charges:", insurance_clean['charges'].isnull().sum())
```
* `dropna()` removes rows containing missing values.
* `subset=['charges']` restricts the removal to rows where charges is missing.
* Other columns may still contain missing values. We will handle those separately.

Output:
```
Dataset shape: (1272, 7)
Missing charges: 0
```

#### Interpretation:

After converting the `charges` column into numeric format, 66 missing values were identified. Since `charges` is the target variable for the regression model, records without a valid target value cannot be used for supervised model training.

Therefore, rows with missing `charges` were removed using `dropna()` with the `charges` column specified as the subset.

**Result:**
* Original dataset: 1,338 rows
* Rows removed: 66
* Remaining dataset: 1,272 rows
* Missing values in `charges`: 0

Missing values in the other feature columns were retained for further cleaning and will be handled separately.

## 2.4 Handle Invalid Values in age
During exploration, we discovered that the age column contains negative values, such as -31, -27 and -64. These are invalid because a person's age cannot be negative.
### 2.4.1 Identify Negative Age Values
What we want to understand
* How many records contain negative ages?
* What is the range of these invalid values?
* Are these records also missing other feature values?
  
#### Step 1: Count negative Values
```
(insurance_clean['age'] < 0).sum()
```
Output: 
```
63
```
There are 63 records with negative ages out of the remaining 1,272 records. This means approximately 4.95% of the dataset contains invalid age values.

#### Step 2: Inspect the affected records
```
insurance_clean[insurance_clean['age'] < 0] 
```
Output: 
| index | age | sex    | bmi    | children | smoker | region    | charges  |
| ----- | --- | ------ | ------ | -------- | ------ | --------- | -------- |
| 5     | -31 | woman  | 25.74  | 0        | no     | southeast | 3756.622 |
| 14    | -27 | male   | 42.13  | 0        | yes    | southeast | 39611.76 |
| 17    | -23 | male   | 23.845 | 0        | no     | northeast | 2395.172 |
| 23    | -34 | female |        | 1        | yes    | null      | 37701.88 |
| 79    | -41 | female | 32.965 | 0        | no     | Northwest | 6571.024 |
| 93    | -35 | male   | 34.77  | 2        | no     | Northwest | 5729.005 |
| 179   | -41 | female | 33.155 | 3        | no     | Northeast | 8538.288 |
| 184   | -44 | null   | 30.69  |          | no     | null      | 7731.427 |
| 196   | -39 | female | 32.8   | 0        | no     | Southwest | 5649.715 |
| 205   | -28 | female | 28.88  | 1        | no     | Northeast | 4337.735 |
| 211   | -40 | male   | 30.875 | 4        | no     | Northwest | 8162.716 |
| 292   | -25 | M      | 45.54  | 2        | yes    | southeast | 42112.24 |
| 315   | -52 | male   | 33.25  | 0        | no     | northeast | 9722.77  |
| 321   | -26 | female | 29.64  | 4        | no     | Northeast | 24671.66 |
| 384   | -44 | male   | 22.135 | 2        | no     | Northeast | 8302.536 |
| 401   | -47 | male   | 47.52  | 1        | no     | southeast | 8083.92  |
| 445   | -45 | female | 33.1   | 0        | no     | Southwest | 7345.084 |
| 457   | -57 | female | 30.495 | 0        | no     | Northwest | 11840.78 |
| 497   | -45 | male   | 28.7   | 2        | no     | southwest | 8027.968 |
| 501   | -43 | male   | 26.03  | 0        | no     | northeast | 6837.369 |
| 563   | -50 | male   | 44.77  | 1        | no     | southeast | 9058.73  |
| 590   | -58 | female | 29     | 0        | no     | Southwest | 11842.44 |
| 599   | -52 | woman  | 37.525 | 2        | no     | Northwest | 33471.97 |
| 616   | -56 | female | 28.595 | 0        | no     | northeast | 11658.12 |
| 620   | -30 | male   | 31.4   | 1        | no     | Southwest | 3659.346 |
| 634   | -51 | male   | 39.7   | 1        | no     | southwest | 9391.346 |
| 655   | -52 | female | 25.3   | 2        | yes    | Southeast | 24667.42 |
| 688   | -47 | female | 24.1   | 1        | no     | southwest | 26236.58 |
| 691   | -47 | male   | 36.2   | -1       | no     | southwest | 8068.185 |
| 715   | -60 | man    | 28.9   | 0        | no     | Southwest | 12146.97 |
| 720   | -51 | female | 40.66  | 0        | no     | northeast | 9875.68  |
| 750   | -37 | F      | 26.4   | 0        | yes    | southeast | 19539.24 |
| 757   | -47 | female | 27.83  | 0        | yes    | southeast | 23065.42 |
| 774   | -41 | man    | 34.2   | 2        | null   | null      | 7261.741 |
| 820   | -45 | man    | 33.7   | 1        | no     | southwest | 7445.918 |
| 828   | -41 | M      | 30.78  | 3        | yes    | Northeast | 39597.41 |
| 849   | -55 | null   | 32.775 |          | null   | null      | 10601.63 |
| 850   | -37 | female | 30.78  | 0        | yes    | northeast | 37270.15 |
| 864   | -51 | male   | 25.4   | 0        | no     | southwest | 8782.469 |
| 879   | -37 | female | 29.5   | 2        | no     | Southwest | 6311.952 |
| 890   | -64 | female | 26.885 |          | null   | Northwest | 29330.98 |
| 927   | -57 | female | 20.1   | 1        | no     | Southwest | 12032.33 |
| 937   | -39 | female | 24.225 | 5        | no     | Northwest | 8965.796 |
| 972   | -22 | woman  | 20.235 | 0        | no     | Northwest | 2527.819 |
| 999   | -36 | woman  | 26.885 | 0        | no     | northwest | 5267.818 |
| 1025  | -21 | female | 34.6   | 0        | no     | Southwest | 2020.177 |
| 1041  | -18 | male   | 23.085 | 0        | no     | Northeast | 1704.7   |
| 1055  | -55 | male   | 27.645 | 0        | no     | Northwest | 10594.5  |
| 1063  | -36 | male   | 28.595 | 3        | no     | northwest | 6548.195 |
| 1142  | -52 | female | 24.86  | 0        | no     | Southeast | 27117.99 |
| 1144  | -50 | male   | 32.3   | 2        | no     | southwest | 9630.397 |
| 1147  | -20 | F      | 31.92  | 0        | no     | northwest | 2261.569 |
| 1150  | -18 | female | 30.305 | 0        | no     | northeast | 2203.736 |
| 1216  | -40 | man    | 25.08  | 0        | no     | Southeast | 5415.661 |
| 1251  | -19 | male   | 19.8   | 0        | no     | southwest | 1241.565 |
| 1254  | -34 | F      | 27.72  | 0        | no     | southeast | 4415.159 |
| 1265  | -64 | male   | 23.76  | 0        | yes    | Southeast | 26926.51 |
| 1273  | -35 | M      | 27.61  | 1        | no     | southeast | 4747.053 |
| 1278  | -39 | male   | 29.925 | 1        | yes    | northeast | 22462.04 |
| 1291  | -19 | male   | 34.9   | 0        | yes    | southwest | 34828.65 |
| 1303  | -43 | male   | 27.8   | 0        | yes    | southwest | 37829.72 |
| 1311  | -33 | female | 26.695 | 0        | no     | northwest | 4571.413 |
| 1334  | -18 | female | 31.92  | 0        | no     | Northeast | 2205.981 |
Some records with negative ages also contain missing values in other columns, such as bmi, sex, smoker, region and children.
This means we need to consider both invalid values and missing values when preparing the dataset.

#### Step 3: Examine the distribution of negative ages
```
insurance_clean.loc[insurance_clean['age'] < 0, 'age'].value_counts().sort_index()
```
Output: 
| age  | count |
|------|-------|
| -64  | 2     |
| -60  | 1     |
| -58  | 1     |
| -57  | 2     |
| -56  | 1     |
| -55  | 2     |
| -52  | 4     |
| -51  | 3     |
| -50  | 2     |
| -47  | 4     |
| -45  | 3     |
| -44  | 2     |
| -43  | 2     |
| -41  | 4     |
| -40  | 2     |
| -39  | 3     |
| -37  | 3     |
| -36  | 2     |
| -35  | 2     |
| -34  | 2     |
| -33  | 1     |
| -31  | 1     |
| -30  | 1     |
| -28  | 1     |
| -27  | 1     |
| -26  | 1     |
| -25  | 1     |
| -23  | 1     |
| -22  | 1     |
| -21  | 1     |
| -20  | 1     |
| -19  | 2     |
| -18  | 3     |

The negative ages range from -18 to -64.
These values are not realistic because a person's age cannot be negative. Interestingly, their absolute values fall within a typical adult age range, but we cannot assume that the negative sign is simply a data-entry mistake that should be reversed.

### 2.4.2 Remove the records with negative ages
We don't have enough information to confirm that -31 actually means 31. Converting them automatically could introduce incorrect information into our training dataset.
```
insurance_clean = insurance_clean[(insurance_clean['age'] >= 0) | (insurance_clean['age'].isnull())]
print("Dataset shape:", insurance_clean.shape)
print("Negative ages remaining:", (insurance_clean['age'] < 0).sum())
```
Output:
```
Dataset shape: (1172, 7)
Negative ages remaining: 0
```

### Interpretation
During data exploration, 63 records containing negative age values were identified. The invalid ages ranged from -18 to -64, accounting for approximately 4.95% of the dataset after removing records with missing `charges`.

Some affected records also contained missing values in other feature columns, including `sex`, `bmi`, `children`, `smoker` and `region`.

Since negative ages are not realistic and there is insufficient evidence to confirm that these values were caused by a reversed sign, they were treated as invalid records.

Records with negative ages were removed from the working dataset to prevent invalid age information from affecting the regression model.

**Result:**
* Records before removal: 1,272
* Records removed: 63
* Remaining records: 1,209
* Negative age values remaining: 0

Records with missing age values were retained for subsequent imputation rather than removed during invalid-value filtering.

## 2.5 Handle Invalid Values in children
During our earlier data exploration, we found that the children column contains negative values. Since the number of children cannot be negative, these values are invalid and need further investigation.
### 2.5.1 Identify Negative Children Values
#### Step 1: Count negative values
```
(insurance_clean['children'] < 0).sum()
```
Output:
```
41
```

#### Step 2: Inspect affected records
```
insurance_clean[insurance_clean['children'] < 0]
```
Output: 
| index | age | sex    | bmi    | children | smoker | region    | charges      |
|-------|-----|--------|--------|----------|--------|-----------|--------------|
| 34    | 28  | male   | 36.4   | -1       | yes    | Southwest | 51194.55914  |
| 66    | 61  | F      | 39.1   | -2       | no     | southwest | 14235.072    |
| 84    | 37  | female | 34.8   | -2       | yes    | Southwest | 39836.519    |
| 105   | 20  | male   | 28.025 | -1       | yes    | Northwest | 17560.37975  |
| 129   | 38  | male   | 34.7   | -2       | no     | southwest | 6082.405     |
| 185   | 36  | male   | 41.895 | -3       | yes    | Northeast | 43753.33705  |
| 225   | 55  | male   | 33.88  | -3       | no     | southeast | 11987.1682   |
| 243   | 40  | male   | 35.3   | -3       | no     | southwest | 7196.867     |
| 267   | 59  | F      | 32.395 | -3       | no     | northeast | 14590.63205  |
| 279   | 51  | female | 21.56  | -1       | no     | Southeast | 9855.1314    |
| 297   | 47  | M      | 25.41  | -1       | yes    | Southeast | 21978.6769   |
| 398   | 64  | male   | 25.6   | -2       | no     | southwest | 14988.432    |
| 431   | 29  | female | 20.235 | -2       | no     | northwest | 4906.40965   |
| 571   | 18  | female | 37.29  | -1       | no     | Southeast | 2219.4451    |
| 578   | 52  | M      | 30.2   | -1       | no     | southwest | 9724.53      |
| 583   | 32  | F      | 23.65  | -1       | no     | southeast | 17626.23951  |
| 621   | 37  | man    | 34.1   | -4       | yes    | Southwest | 40182.246    |
| 645   | 48  | male   | 30.78  | -3       | no     | northeast | 10141.1362   |
| 653   | 45  | female | 36.3   | -2       | no     | southeast | 8527.532     |
| 662   | 32  | female | 31.54  | -1       | no     | Northeast | 5148.5526    |
| 667   | 40  | female | 32.775 | -2       | yes    | Northwest | 40003.33225  |
| 669   | 40  | female | 29.81  | -1       | no     | southeast | 6500.2359    |
| 811   | 33  | female | 42.94  | -3       | no     | northwest | 6360.9936    |
| 853   | 53  | female | 23.75  | -2       | no     | northeast | 11729.6795   |
| 938   | 18  | male   | 26.18  | -2       | no     | southeast | 2304.0022    |
| 941   | 50  | F      | 46.09  | -1       | no     | Southeast | 9549.5651    |
| 962   | 63  | female | 35.2   | -1       | no     | southeast | 14474.675    |
| 968   | 21  | male   | 25.745 | -2       | no     | northeast | 3279.86855   |
| 994   | 27  | female | 20.045 | -3       | yes    | Northwest | 16420.49455  |
| 1005  | 29  | male   | 31.73  | -2       | no     | northwest | 4433.3877    |
| 1009  | 51  | man    | 27.74  | -1       | no     | northeast | 9957.7216    |
| 1017  | 26  | female | 34.2   | -2       | no     | southwest | 3987.926     |
| 1021  | 22  | female | 31.02  | -3       | yes    | southeast | 35595.5898   |
| 1079  | 63  | male   | 33.66  | -3       | no     | Southeast | 15161.5344   |
| 1083  | 32  | male   | 31.5   | -1       | no     | Southwest | 4076.497     |
| 1094  | 50  | female | 33.7   | -4       | no     | southwest | 11299.343    |
| 1095  | 18  | woman  | 31.35  | -4       | no     | Northeast | 4561.1885    |
| 1128  | 34  | male   | 32.8   | -1       | no     | southwest | 14358.36437  |
| 1191  | 41  | woman  | 21.755 | -1       | no     | northeast | 13725.47184  |
| 1224  | 41  | male   | 23.94  | -1       | no     | Northeast | 6858.4796    |
| 1299  | 19  | female | 25.745 | -1       | no     | northwest | 2710.82855   |

#### Step 3: Check the distribution of negative values
```
insurance_clean.loc[insurance_clean['children'] < 0, 'children'].value_counts().sort_index()
```
Output:
| children | count |
|----------|-------|
| -4       | 3     |
| -3       | 9     |
| -2       | 12    |
| -1       | 17    |

From the affected records, we can observe:
1. Negative values range from -1 to -4.
2. The affected records contain a mixture of ages, genders, smoking statuses and regions.
3. There is no obvious pattern indicating that the negative values should simply be converted into positive values.
4. The actual number of children cannot be determined from the other columns.

### 2.5.2 Remove Records with Negative Children Values
We should not convert -1 to 1, for example, because we have no evidence that the negative sign is simply a data-entry mistake.
```
insurance_clean = insurance_clean[(insurance_clean['children'] >= 0) |(insurance_clean['children'].isnull())]
print("Dataset shape:", insurance_clean.shape)
print("Negative children values:", (insurance_clean['children'] < 0).sum())
print("Missing children values:", insurance_clean['children'].isnull().sum())
```
Output: 
```
Dataset shape: (1168, 7)
Negative children values: 0
Missing children values: 29
```

### Interpretation
During data exploration, 41 records containing negative values in the `children` column were identified. The invalid values ranged from -1 to -4.

The affected records contained a mixture of ages, genders, smoking statuses and regions. However, there was no clear pattern to determine the actual number of children. Therefore, these values were treated as invalid rather than converted into positive values.

Records with negative `children` values were removed from the working dataset to prevent invalid information from affecting the regression model. Missing values were retained for further handling during the missing value treatment stage.

Result:
* Records before removal: 1,209
* Records removed: 41
* Remaining records: 1,168
* Negative children values remaining: 0
* Missing children values retained: 29

## 2.6 Standardize Categorical Values
During our earlier data exploration, we found inconsistent values in the categorical columns. For example, Male, male, M and man represent the same category but are recorded differently. These inconsistencies need to be standardized so the model can interpret the categories correctly.
### 2.6.1 Sex
#### Step 1: Inspect current values
```
insurance_clean['sex'].value_counts(dropna=False)
```
Output: 
| sex    | count |
|--------|-------|
| male   | 469   |
| female | 451   |
| M      | 58    |
| woman  | 55    |
| man    | 55    |
| F      | 51    |
| null   | 29    |

#### Step 2: Inspect unique values
```
insurance_clean['sex'].unique()
```
Output:
```
insurance_clean['sex'].unique()
array(['female', 'male', 'woman', 'F', 'man', 'M', nan], dtype=object)
```

### 2.6.2 Standardize the sex column
```
sex_mapping = {
    'male': 'male',
    'M': 'male',
    'man': 'male',
    'female': 'female',
    'F': 'female',
    'woman': 'female'
}

insurance_clean['sex'] = insurance_clean['sex'].map(sex_mapping)
insurance_clean['sex'].value_counts(dropna=False)
```
Output:
| sex    | count |
|--------|-------|
| male   | 582   |
| female | 557   |
| null   | 29    |

### 2.6.3 Smoker
#### Step 1: Check value counts
```
insurance_clean['smoker'].value_counts(dropna=False)
```
Output: 
| smoker | count |
|--------|-------|
| no     | 904   |
| yes    | 233   |
| null   | 31    |

#### Step 2: Check unique values
```
insurance_clean['smoker'].unique()
```
Output: 
```
array(['yes', 'no', nan], dtype=object)
```
> The column contains only two valid categories. There are no inconsistent variations such as Yes, YES or No. Therefore, no standardization is required for this column.

### 2.6.4 Region
#### Step 1: Check value counts
```
insurance_clean['region'].value_counts(dropna=False)
```
Output:
| region    | count |
|-----------|-------|
| Southeast | 157   |
| southeast | 147   |
| Northwest | 144   |
| southwest | 140   |
| northwest | 140   |
| Northeast | 140   |
| northeast | 137   |
| Southwest | 132   |
| null      | 31    |


#### Step 2: Check unique values
```
insurance_clean['region'].unique()
```
Output:
```
array(['southwest', 'Southeast', 'southeast', 'northwest', 'Northwest',
       'Northeast', 'northeast', 'Southwest', nan], dtype=object)
```

### 2.6.5 Standardize the region column
```
insurance_clean['region'] = insurance_clean['region'].str.title()
insurance_clean['region'].value_counts(dropna=False)
```
This converts the first letter of each region to uppercase while keeping the remaining letters lowercase. Missing values will remain unchanged.

Output:
| region    | count |
|-----------|-------|
| Southeast | 304   |
| Northwest | 284   |
| Northeast | 277   |
| Southwest | 272   |
| null      | 31    |

### Interpretation
During data exploration, inconsistent categorical values were identified in the sex and region columns. The sex column contained different representations of male and female, while the region column contained inconsistent capitalization for the four regions.

The `.map()` method was used with a mapping dictionary to standardize the sex column into two categories, male and female. Meanwhile, the `.str.title()` method was applied to the region column to ensure consistent capitalization.

The smoker column was also inspected and found to contain only two consistent categories, yes and no. Therefore, no changes were required.

Missing values in all three columns were preserved for further handling during the missing value treatment stage.

Result:
* **sex**: 582 male, 557 female, 29 missing values.
* **smoker**: 904 no, 233 yes, 31 missing values.
* **region**: 304 Southeast, 284 Northwest, 277 Northeast, 272 Southwest, 31 missing values.
All categorical values are now consistently represented, making the dataset more suitable for further preprocessing and regression modelling.

## 2.7 Handle Missing Feature Values
So far, we have removed records with missing charges because charges is our target variable. However, we have retained missing values in the feature columns because we may still be able to use those records for model training.

Next step is to understand how many missing values remain and decide how to handle them.

### 2.7.1 Identify Missing Values
#### Step 1: Check missing value counts
```
insurance_clean.isnull().sum()
```
This shows the number of missing values in each column.

Output:
| column   | missing values |
|----------|----------------|
| age      | 37             |
| sex      | 29             |
| bmi      | 37             |
| children | 29             |
| smoker   | 31             |
| region   | 31             |
| charges  | 0              |

#### Step 2: Calculate missing value percentages
```
insurance_clean.isnull().sum() / len(insurance_clean) * 100
```
This helps us understand the proportion of missing values in each column.

Output: 
| column   | missing percentage |
|----------|--------------------|
| age      | 3.1678%            |
| sex      | 2.4829%            |
| bmi      | 3.1678%            |
| children | 2.4829%            |
| smoker   | 2.6541%            |
| region   | 2.6541%            |
| charges  | 0%                 |

> Since the missing percentages are relatively low, we can consider imputation instead of removing more records. However, we should not automatically fill every missing value with the mean or mode. We need to choose a suitable method for each column.

### 2.7.2 Determine the Appropriate Imputation Method
Before filling the missing values, let's inspect the distribution of each feature to decide which imputation method is appropriate.

#### Step 1: Inspect numerical columns
```
insurance_clean[['age', 'bmi', 'children']].describe()
```
We'll use the results to decide whether to use the mean or median.
| Column     | Possible treatment | Reason                                      |
|------------|--------------------|---------------------------------------------|
| `age`      | Median             | Less affected by extreme values             |
| `bmi`      | Median             | More robust if the distribution is skewed    |
| `children` | Median             | Preserves a typical whole-number value       |

Output:
| Statistic | age          | bmi          | children |
|-----------|--------------|--------------|----------|
| count     | 1131         | 1131         | 1139     |
| mean      | 39.2687886826 | 30.5860831123 | 1.0570676032 |
| std       | 14.1764571207 | 6.1360131401  | 1.1928649205 |
| min       | 18           | 15.96        | 0        |
| 25%       | 26           | 26.22        | 0        |
| 50%       | 39           | 30.21        | 1        |
| 75%       | 51.5         | 34.6375      | 2        |
| max       | 64           | 53.13        | 5        |

Step 2: Inspect categorical columns
```
for column in ['sex', 'smoker', 'region']:
    print(f"\n{column.upper()}")
    print(insurance_clean[column].value_counts())
```
We'll use the mode (most frequent category) as a possible treatment for missing categorical values.

Output:
```
SEX
male      582
female    557
Name: sex, dtype: int64

SMOKER
no     904
yes    233
Name: smoker, dtype: int64

REGION
Southeast    304
Northwest    284
Northeast    277
Southwest    272
Name: region, dtype: int64
```

### 2.7.3 Selected Imputation Methods
Based on the findings, we will use the following methods:
| Column     | Data type  | Imputation method | Replacement value |
|------------|------------|-------------------|-------------------|
| `age`      | Numerical  | Median            | 39                |
| `bmi`      | Numerical  | Median            | 30.21             |
| `children` | Numerical  | Median            | 1                 |
| `sex`      | Categorical | Mode             | male              |
| `smoker`   | Categorical | Mode             | no                |
| `region`   | Categorical | Mode             | Southeast         |

### Interpretation
The dataset contains missing values across all six feature columns, with missing percentages ranging from approximately 2.48% to 3.17%.

For numerical columns (age, bmi and children), median imputation was selected because the median is less sensitive to extreme values. For categorical columns (sex, smoker and region), mode imputation was selected to replace missing values with the most frequently occurring category.

The missing values will be handled during model preprocessing using statistics calculated from the training data only to prevent data leakage.

#### Important: Should we fill the missing values now?
Not yet. Since we will evaluate our regression model using training and testing data, we should calculate imputation values using the training data only.

Otherwise, information from the testing data could influence the training process. This is known as data leakage.

Later, we will use Scikit-learn's `SimpleImputer` to handle missing values as part of our model preprocessing pipeline.

## 2.8 Final Data Quality Verification
### 2.8.1 Check Dataset Structure
#### Step 1: Check the dataset shape
```
print("Dataset shape:", insurance_clean.shape)
```
This tells us how many records and columns remain after cleaning.

Output:
```
Dataset shape: (1168, 7)
```

#### Step 2: Check data types and non-null counts
```
insurance_clean.info()
```
This helps us verify that:
* age, bmi, children and charges have numerical data types.
* sex, smoker and region have appropriate categorical data types.
* charges contains no missing values.

Output:
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1168 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1131 non-null   float64
 1   sex       1139 non-null   object 
 2   bmi       1131 non-null   float64
 3   children  1139 non-null   float64
 4   smoker    1137 non-null   object 
 5   region    1137 non-null   object 
 6   charges   1168 non-null   float64
dtypes: float64(4), object(3)
memory usage: 73.0+ KB
```

### 2.8.2 Verify Remaining Invalid Values
#### Step 1: Check negative values
```
print("Negative ages:", (insurance_clean['age'] < 0).sum())
print("Negative children:", (insurance_clean['children'] < 0).sum())
```
Output:
```
Negative ages: 0
Negative children: 0
```

#### Step 2: Check categorical values
```
for column in ['sex', 'smoker', 'region']:
    print(f"\n{column.upper()}")
    print(insurance_clean[column].value_counts(dropna=False))
```
Output:
```
SEX
male      582
female    557
NaN        29
Name: sex, dtype: int64

SMOKER
no     904
yes    233
NaN     31
Name: smoker, dtype: int64

REGION
Southeast    304
Northwest    284
Northeast    277
Southwest    272
NaN           31
Name: region, dtype: int64
```

### 2.8.3 Final Dataset Summary
```
print("Final dataset shape:", insurance_clean.shape)

print("\nTotal missing values:")
print(insurance_clean.isnull().sum().sum())

print("\nDuplicate rows:")
print(insurance_clean.duplicated().sum())

print("\nData types:")
print(insurance_clean.dtypes)
```
Output:
```
Final dataset shape: (1168, 7)

Total missing values:
194

Duplicate rows:
1

Data types:
age         float64
sex          object
bmi         float64
children    float64
smoker       object
region       object
charges     float64
dtype: object
```

### 2.8.4 Investigate Duplicate Records
#### Step 1: Display duplicate records
```
insurance_clean[insurance_clean.duplicated(keep=False)]
```
The `keep=False` argument displays all records involved in duplication, rather than showing only the later occurrence.

Output:
```
| index | age | sex  | bmi   | children | smoker | region    | charges   |
|-------|-----|------|-------|----------|--------|-----------|-----------|
| 195   | 19  | male | 30.59 | 0        | no     | Northwest | 1639.5631 |
| 581   | 19  | male | 30.59 | 0        | no     | Northwest | 1639.5631 |
```

#### Step 2: Check the duplicate count
```
insurance_clean.duplicated().sum()
```
We need to determine whether the records are exact duplicates across all seven columns.

Output:
```
1
```
