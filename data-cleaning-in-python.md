# Data Cleaning in Python
Cleaning your data is **a process of removing errors, outliers, and inconsistencies** and ensuring that all of your data is in a format that is appropriate for your analysis. Data that contains many errors or that hasn’t gone through this data-cleaning process is referred to as dirty data.

## Understanding Dirty Data
### Missing values
Incomplete data sets are extremely common. Your dataset may be missing several years of data, only have some information on a customer, or not contain your company’s full complement of products.
Missing values can have multiple effects on your analysis. Large portions of crucial data missing can cause bias in your results. Additionally, NaN values or missing cells in a DataFrame **may break some Python code**, causing a lot of frustration while building your model.

### Outliers
Outliers are **values that are far outside the norm and not representative of the data**. Outliers can be the result of a typo or exceptional circumstances. It is important to differentiate true outliers from informative extreme situations. Outliers can skew your results, ultimately suggesting the wrong answer.

### Duplicates
Duplicate data entries can **overrepresent one entry in your analysis**, leading to the wrong conclusion. Be aware of duplicates that occur more than once and those that contain conflicting or updated information.

### Errorneous data
Sometimes, the **values in our data set are simply wrong**. You may have the wrong spelling for a customer’s name, the wrong product number, outdated information, or incorrectly labeled data.
It can sometimes be challenging to determine if your data is erroneous, which is why verification of your source is so important! Remember, your analysis is only as good as your data.

### Inconsistencies
Inconsistencies come in many forms. There can be **inconsistent data entries**, which may point to a typo or an error. If you see a customer age backward, an ingredient that changes ID number, or a product with two simultaneous prices, it is worth a closer look to make sure everything is correct.

Another problematic type of inconsistency is inconsistency in the format of the data. Different values may be reported in:
* different **units** (kilometers vs. miles vs. inches)
* different **styles** (Month Day, Year vs. Day-Month-Year)
* different **data types** (floats vs. integers)
* different **file types** (.jpg vs. .png)

These inconsistencies will make it challenging, if not impossible, for your code to interpret the values correctly. This may result in an incorrect analysis or in your code not running at all.

## Data Exploration and Preprocessing
It is crucial that you understand your dataset before you use it in a complex analysis. To develop this kind of understanding of your data, you should do some data exploration. You can think of this step as the pre-analysis analysis.

Let’s look at some typical steps for conducting a data exploration on a table-formatted dataset:
1. First and foremost, **look at the source of your dataset** and determine if that source has any bias or agenda that may affect the quality or reliability of your data.
2. Learn **the context of your data** and any other factors that may have affected your data that aren’t internally accounted for.
3. Determine **how many different variables you have**. In a table-formatted dataset, the variables are typically the columns, while each data entry is a row.
4. Determine **how many different categories within each variable you have**. For example, if one of your variables is Type of Fruit, you should have a good idea of how many different types of fruit are represented within this variable.
5. Look at the **summary statistics for each column**, including the mean, median, variance, and standard deviation.
6. If possible, **plot each variable and as many pairs of variables as you can** and really look at the plots. Look for any skews, outliers, trends, or correlations that may be worth looking into further.
7. If your dataset is part of a relational database, **look closely at the relationships** and make sure you understand how the different tables relate to each other.
8. If applicable, use the **profiling feature of the pandas library** to generate a profile report. This will provide you with valuable information about your dataset.
https://www.geeksforgeeks.org/python/pandas-profiling-in-python/ 

### Why exploration and preprocessing matters
This all may seem superfluous to your intended analysis, but there are a few important reasons you should always take these steps. Firstly, doing this will give you a **great understanding of the limits of your dataset**, which is essential if you are to trust the results of your final analyses.

Secondly, data exploration can **point you in the direction of important trends and analyses you hadn’t considered previously**. These have the potential to add to your intended analyses or to present a complicating factor that you need to take into account.

Thirdly, this pre-analysis can **be your first clue to where you may have dirty data**. This can be the first time you see that outlier, realize there are twice as many categories as there should be, or discover that the data collection method was different last year than it was the year before. These are all critical pieces of information that should encourage you to be curious about your dataset.

## Techniques for Data Cleaning in Python
### Handling missing values
Invariably, when you have a large dataset, some fields will be missing in one or more entries. Not only does this mean you are missing valuable data, but the NaN entries can trip up some Python functions, making your model inaccurate.

When you find a missing value, you have a choice to either **eliminate that entry from analysis** or try to **impute a reasonable value to put in its place**. If your dataset is reasonably small, I would suggest viewing the rows that contain missing values so you can determine the best course of action.

```
import pandas as pd # Identify rows with NaN values
rows_with_nan = df[df.isnull().any(axis=1)] # View the rows with NaN values
print(rows_with_nan)
```

Deleting an entry is typically not the first choice, as it means taking potentially valuable information out of the analysis.

However, **sometimes deletion is the best option**, for example, if the entry doesn’t provide enough other information to warrant keeping it (like if the entry is only a date with no valuable information).

Deletion is also the easiest way to deal with entries that contain missing values. So, if you are short on time and the rest of the entry is not important to the analysis, deleting entries with NaNs is an option.

You can use the **_.dropna()_** method from the Pandas library to drop rows from a DataFrame that contain NaNs.

```
import pandas as pd # Assuming df is your DataFrame
df.dropna(inplace=True)
```

The generally preferred method of handling missing values is imputing a reasonable value. Finding a reasonable value to impute can be an art in its own right, but there are several established methods that can be a good starting point.

The **_.fillna()_** method from Pandas will impute missing values using the mean so as not to change the distribution.

```
import pandas as pd # Assuming df is your DataFrame
dr.fillna(df.mean(), inplace=True) # Replace NaN values with the mean of the column
```

The scikit-learn library has a simple imputation function that works well, too.
```
from sklearn.impute import SimpleImputer
import pandas as pd # Assuming df is your DataFrame
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(impute.fit_transform(df), columns=df.columns)
```

