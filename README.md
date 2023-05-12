# Ex-06-Feature-Transformation
## AIM
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM

STEP 1:

Read the given Data

STEP 2:

Clean the Data Set using Data Cleaning Process

STEP 3:

Apply Feature Transformation techniques to all the features of the data set

STEP 4:

Print the transformed features

## PROGRAM
```
NAME: SHARANGINI T K
REG NO: 212222230143

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

# READ CSV FILES
df=pd.read_csv("/content/Data_to_Transform.csv")
df
# BASIC PROCESS
df.head()

df.info()

df.describe()

df.tail()

df.shape

df.columns

df.isnull().sum()

df.duplicated()

# LOG TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# MODERATE POSITIVE SKEW
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

# RECIPROCAL TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# SQUARE ROOT TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# POWER TRANSFORMATION
# MODERATE POSITIVE SKEW
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")

df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

# QUANTILE TRANSFORMATION
# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')

df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

```
# OUTPUT
# Importing Libraries
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/7f639240-48be-4c7c-ba3e-b9da66c60d5a)
# Reading CSV File
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/bb277d2b-e8f6-45c0-9a8a-3697fc63611a)
# Basic Process
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/44863674-57e5-4f06-b131-77ee6b620a41)
# Before Transformation
## Highly Positive Skew
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/dfce113a-4173-415e-a0bb-b11a0998a796)
## Highly Negative Skew
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/bd727113-44f3-40ef-bb0c-8c344d748304)
## Moderate Positive Skew
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/2cdd0b8f-41b9-4fd1-82f7-e561893b29f4)
## Moderate Negative Skew
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/c6ab2833-fc6e-476d-9dbf-3870a2cb72f6)
# Log Transformation
## Highly Positive Skew
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/85daa472-ca85-4e36-830c-10be61ef0dba)
## Moderate Positive Skew
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/07ea6417-924b-4b9e-bc97-f3afb3ca7e23)
# Reciprocal Transformation
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/a1f8ceab-8210-4143-8b0d-657a7ba23c2f)
# Square Root Transformation
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/94724407-1bab-4ebb-952c-2cd2fc8f633b)
# Power Transformation
## Moderate Positive Skew
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/c5b445fa-effd-4633-940b-7a44a9a55506)
## Moderate Negative Skew
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/5e90dc0b-c74e-4000-9fde-3a8bb4061d13)
# Quantile Transformation
## Moderate Negative Skew
![image](https://github.com/shara56/Ex-06-Feature-Transformation/assets/113497104/33fad630-b58d-4b09-9844-7c934e7be890)

## RESULT
Thus feature transformation is done for the given dataset.
