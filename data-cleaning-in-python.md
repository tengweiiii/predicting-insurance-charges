# Data Cleaning in Python
##### Cleaning your data is a process of removing errors, outliers, and inconsistencies and ensuring that all of your data is in a format that is appropriate for your analysis. Data that contains many errors or that hasn’t gone through this data-cleaning process is referred to as dirty data.

## Understanding Dirty Data
### Missing values
---
##### Incomplete data sets are extremely common. Your dataset may be missing several years of data, only have some information on a customer, or not contain your company’s full complement of products.
##### Missing values can have multiple effects on your analysis. Large portions of crucial data missing can cause bias in your results. Additionally, NaN values or missing cells in a DataFrame may break some Python code, causing a lot of frustration while building your model.

### Outliers
---
##### Outliers are values that are far outside the norm and not representative of the data. Outliers can be the result of a typo or exceptional circumstances. It is important to differentiate true outliers from informative extreme situations. Outliers can skew your results, ultimately suggesting the wrong answer.

### Duplicates
---
##### Duplicate data entries can overrepresent one entry in your analysis, leading to the wrong conclusion. Be aware of duplicates that occur more than once and those that contain conflicting or updated information.

### Errorneous data
---
##### Sometimes, the values in our data set are simply wrong. You may have the wrong spelling for a customer’s name, the wrong product number, outdated information, or incorrectly labeled data.
##### It can sometimes be challenging to determine if your data is erroneous, which is why verification of your source is so important! Remember, your analysis is only as good as your data.

### Inconsistencies
---
##### Inconsistencies come in many forms. There can be inconsistent data entries, which may point to a typo or an error. If you see a customer age backward, an ingredient that changes ID number, or a product with two simultaneous prices, it is worth a closer look to make sure everything is correct.
##### Another problematic type of inconsistency is inconsistency in the format of the data. Different values may be reported in:
* different **units** (kilometers vs. miles vs. inches)
* different **styles** (Month Day, Year vs. Day-Month-Year)
* different **data types** (floats vs. integers)
* different **file types** (.jpg vs. .png)
##### These inconsistencies will make it challenging, if not impossible, for your code to interpret the values correctly. This may result in an incorrect analysis or in your code not running at all.
