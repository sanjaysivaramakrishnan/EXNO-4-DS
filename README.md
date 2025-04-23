# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
**Feature Scaling**
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv(r"C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_to_datascience\data_sets\bmi.csv")
df.head()
df.dropna()
# TYPE CODE TO FIND MAXIMUM VALUE FROM HEIGHT AND WEIGHT FEATURE
max_vals=np.max(np.abs(df[['Height']]))
max_vals1=np.max(np.abs(df[['Weight']]))
print("Height =",max_vals)
print("Weight =",max_vals1)
from sklearn.preprocessing import MinMaxScaler
#Perform minmax scaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
from sklearn.preprocessing import StandardScaler
df1=df
#Perform standard scaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
from sklearn.preprocessing import Normalizer
#Perform Normalizer
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
from sklearn.preprocessing import MaxAbsScaler
#Perform MaxAbsScaler
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
from sklearn.preprocessing import RobustScaler
#Perform RobustScaler
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
df=pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_to_datascience\data_sets\titanic_dataset.csv')
df.columns
df.shape
X = df.drop(columns="Survived",axis=1)       # feature matrix
y = df['Survived']
df1=df
#drop the following columns -"Name", "Sex", "Ticket", "Cabin", "Embarked" and store it in df1
df1=df1.drop(columns=['Name','Sex','Ticket','Cabin','Embarked'])
df1.columns
df1['Age'].isnull().sum()
#fill null values of age column using forward fill method
df1['Age']=df1['Age'].ffill()
df1['Age'].isnull().sum()
df1.isnull().sum()
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns
X=df1.iloc[:,0:6]
y=df1.iloc[:,6]
X.columns
y=y.to_frame()
y.columns
X = df1.drop(columns="Survived",axis=1)       # feature matrix
y = df1['Survived']
feature.fit(X,y)
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv(r"C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_to_datascience\data_sets\income.csv",na_values=[ " ?"])
data

data.isnull().sum()

missing=data[data.isnull().any(axis=1)]
missing

data2=data.dropna(axis=0)
data2
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs


data2
new_data=pd.get_dummies(data2, drop_first=True)
new_data

columns_list=list(new_data.columns)
print(columns_list)


features=list(set(columns_list)-set(['SalStat']))
print(features)
y=new_data['SalStat'].values
print(y)

x=new_data[features].values
print(x)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

data.shape

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

tips.time.unique()

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```

![image](https://github.com/user-attachments/assets/2164c075-da07-49cf-8e99-5570c08fecdd)
![image](https://github.com/user-attachments/assets/6016a7b5-b0f9-4f81-8184-9798c8d9161c)
![image](https://github.com/user-attachments/assets/368cb342-78e9-4867-932f-dc43f3b94089)
![image](https://github.com/user-attachments/assets/a4f6a31b-ede0-4503-8aad-63dabedc175e)
![image](https://github.com/user-attachments/assets/08f279b2-da20-45dd-9e87-5cd3d7587420)
![image](https://github.com/user-attachments/assets/4622ccf0-a570-4335-8979-5ee723cc3c06)
![image](https://github.com/user-attachments/assets/b12096fe-4ee3-4bbb-bd8d-d69d96a15a9f)
![image](https://github.com/user-attachments/assets/30c5a33f-0084-4e75-b19b-8ac42c5985ae)
![image](https://github.com/user-attachments/assets/cd1e4517-a349-4d1c-aa83-a9b8d19362e4)
![image](https://github.com/user-attachments/assets/d797eb53-f27d-4fcf-8174-a8958896d012)
![image](https://github.com/user-attachments/assets/5a2d4087-6950-4a25-8170-bdaac232a6e2)
![image](https://github.com/user-attachments/assets/6b0f1ec5-1f16-4565-b70b-9263ff16cec0)


# RESULT:
       # INCLUDE YOUR RESULT HERE
