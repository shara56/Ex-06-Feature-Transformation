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

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
## OUTPUT
![image](https://user-images.githubusercontent.com/113497104/232674783-77a7b0b1-b2f9-4892-a00d-a705984695c3.png)

![image](https://user-images.githubusercontent.com/113497104/232674807-8695443b-6b4d-4568-a9cc-73defed0adb0.png)

![image](https://user-images.githubusercontent.com/113497104/232674833-a48fc7a8-bbcf-40b5-bc37-8ff96beb3808.png)

![image](https://user-images.githubusercontent.com/113497104/232674867-7fbf5560-772a-4b57-8a93-6eb7b03f8c17.png)

![image](https://user-images.githubusercontent.com/113497104/232674890-8b417e74-5148-40b5-bc6b-c0fb5250aabd.png)

![image](https://user-images.githubusercontent.com/113497104/232674916-6dfd73bd-a10e-4075-840f-7ad48cad622b.png)

![image](https://user-images.githubusercontent.com/113497104/232674941-8e3ef62b-0e22-4c56-aed9-ff03483fa3fd.png)

![image](https://user-images.githubusercontent.com/113497104/232674953-8e15dc8b-8c34-405e-9e92-654db19b5ef5.png)

![image](https://user-images.githubusercontent.com/113497104/232674975-619687b9-4fd7-448e-b366-6eaea758fb04.png)

![image](https://user-images.githubusercontent.com/113497104/232674984-7dbbb470-2c8c-4eb4-9a8a-dff8c6f29aef.png)

## RESULT

Thus feature transformation is done for the given dataset.
