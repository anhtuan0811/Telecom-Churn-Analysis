# 🛒 Case Study - Predictors of telecom churn

<p align="right"> Using Python - Google Colab </p>


## :books: Table of Contents <!-- omit in toc -->

- [🔢 PYTHON - GOOGLE COLAB]
  - [Import Library and dataset](#-import-library-and-dataset)
  - [Overall Information ](#1%EF%B8%8F⃣-overall-information)
  - [Data Cleaning](#2%EF%B8%8F⃣-data-cleaning)
  - [Data exploration](#3%EF%B8%8F⃣--data-exploration)
  - [Fitting](#4%EF%B8%8F⃣-fitting-model)
  - [Tuning](#5%EF%B8%8F⃣-tuning)
  - [Evaluate Model](6%EF%B8%8F⃣-evaluate-models)
  - [Pickling Model](#7%EF%B8%8F⃣-pickling-the-model)


---

## 👩🏼‍💻 PYTHON - GOOGLE COLAB

### 🔤 IMPORT LIBRARY AND DATASET 

<details><summary> Click to expand code </summary>
  
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
%matplotlib inline

```

```python
#import dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
```
  
</details>

---
### 1️⃣ Overall information
<details><summary> Click to expand code </summary>

```python
df.head() 
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/6fd98ca4-0510-41e5-911c-d2393ac1df07)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/e4e71f00-47b5-45dc-93ab-2e053b50ab4c)
 
```python
df.info()
```  
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/ae4c6f5f-3f1a-49cd-8a14-41b7de22a4ac)
<br> Here, we don't have any missing data.

```
  
```python
df.describe()
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/3fed8325-458a-4b33-b732-2ea476d434d1)

- SeniorCitizen is actually a categorical variable, hence the 25%-50%-75% distribution is not proper.
- 75% of customers have tenure less than 55 months.
- The average Monthly charges are USD 64.76, whereas 25% of customers pay more than USD 89.85 per month.


</details>

---

### 2️⃣ Data Cleaning
<details><summary>  2.1. Create a copy of base data for manupulation & processing </summary>

```python
df1 = df.copy()

```

</details>

<details><summary>  2.2. Convert Total Charges to numerical data type </summary>

```python
df1.TotalCharges = pd.to_numeric(df1.TotalCharges, errors='coerce')
df1.isnull().sum()

```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/91eaa372-1ae7-4c76-a00e-3b5d26c45d1d)
<br>  As we can see there are 11 missing values in TotalCharges column. Since the % of these records compared to total dataset is very low ie 0.15%, it is safe to ignore them from further processing.

```python
df1.dropna(how = 'any', inplace = True)
```
</details>
<details><summary>  2.3. Divide customers into bins based on tenure </summary>

```python
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

df1['tenure_group'] = pd.cut(df1.tenure, range(1, 80, 12), right=False, labels=labels)
df1['tenure_group'].value_counts()
```

![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/58a0e666-d31c-4913-82e6-e4985dda9ebd)



</details>
<details><summary>  2.4. Remove columns not required for processing  </summary>

```python
df1.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
df1.head()

```
</details>

---

### 3️⃣  Data exploration


<details><summary> Churn  </summary>

 ```python
# Churn
df['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02)

```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/e173a944-49b8-47eb-b17e-af775ff9d1e1)
```python

100*df['Churn'].value_counts()/len(df['Churn'])
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/629fb40c-b710-44ea-b2b5-2762abeb3311)

<br>
--> In terms of the number of 'yes' and 'no' responses, Data is highly imbalanced, ratio = 73:27
</details>

<details><summary> Plot distibution of individual predictors by churn </summary> 
  
```python
for i, predictor in enumerate(df1.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i, figsize=(10, 6))
    sns.countplot(data=df1, x=predictor, hue='Churn')
  
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/518604d6-3b40-4358-ac64-bd6bc90bd641)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/9599cc59-1524-46c1-8794-3b957fde7774)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/3753f60a-445c-4659-9ee3-2f4d220efcee)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/c35463c7-e707-4642-8dff-eab7a8e9ac30)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/a9a39d24-f9b0-4672-8c35-6e149a45a174)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/b90a1827-0661-4a69-bcea-3726c9be53aa)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/d95cc63f-3cbd-4c03-b66b-c9ddb154d553)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/f6465c8c-ff10-48e1-af8f-e18122243556)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/8b1a502d-cca9-474d-91c2-f8cbf78d97f3)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/a91df331-285c-47b5-8b7b-a8d5f7a57768)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/f9fac884-9880-4222-b72f-8c92fb9d8b38)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/56666805-5fbc-401e-95ac-9e0f4e789f26)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/16f4689f-69df-4bd9-8f6e-1a8175dd1bde)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/ded64753-a7ab-495f-bb6e-38e672db7a9c)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/8db41eab-bc6d-4e83-9648-c69bb249f92f)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/b4854e77-9e25-4e7e-b409-a99af8a346f9)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/71c37f51-a96b-44af-b7f1-4a64db363bb8)


</details>
 
<details><summary> Convert the target variable 'Churn' in a binary numeric variable  </summary> 

```python
df1['Churn'] = np.where(df1.Churn == 'Yes',1,0)
```

</details>

<details><summary> Convert all the categorical variables into dummy variables  </summary> 
  
```python
df1_dummies = pd.get_dummies(df1)
df1_dummies.head()
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/5defe553-2954-4c19-a455-24c49e1803c4)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/68b490b5-8830-40b7-be3b-bc25d31517f2)

</details>

<details><summary> Relationship between Monthly Charges and Total Charges </summary> 
  
```python
sns.lmplot(data=df1_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/5c2d64ed-fafc-477a-a969-b7bad9b331e4)

<br>

--> Total Charges increase as Monthly Charges increase - as expected.
</details>

<details><summary> Churn by Monthly Charges and Total Charges  </summary> 


```python

Tot = sns.kdeplot(df1_dummies.TotalCharges[(df1_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(df1_dummies.TotalCharges[(df1_dummies["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')

```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/ec471585-6f9c-460c-8f1a-16fdd43b3ecb)

```python

Mth = sns.kdeplot(df1_dummies.MonthlyCharges[(df1_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(df1_dummies.MonthlyCharges[(df1_dummies["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/e8fd34d9-c170-4044-99d5-c97b09dc1d43)

<br>

--> Churn is high when Monthly Charges are high. Higher Churn at lower Total Charges. Nonetheless, when we merge the findings of three variables, specifically Tenure, Monthly Charges, and Total Charges, the situation becomes more evident. A situation with higher Monthly Charges and shorter tenure leads to lower Total Charges. As a result, all three elements, namely elevated Monthly Charges, reduced tenure, and decreased Total Charges, are associated with a heightened churn rate.
</details>
<details><summary> Build a correlation of all predictors with 'Churn'  </summary> 

```python
plt.figure(figsize=(20,8))
df1_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/ca7f3d6f-1f13-4cd0-ad2c-ec4e50b1ea55)

- HIGH Churn seen in case of Month to month contracts, No online security, No Tech support, First year of subscription and Fibre Optics Internet
- LOW Churn is seens in case of Long term contracts, Subscriptions without internet service and The customers engaged for 5+ years
- Factors like Gender, Availability of PhoneService and Number of multiple lines have alomost NO impact on Churn

</details>
<details><summary> Univariate Analysis </summary> 

```python
new_df1_target0=df1.loc[df1["Churn"]==0]
new_df1_target1=df1.loc[df1["Churn"]==1]

def uniplot(df,col,title,hue =None):

    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30


    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright')

    plt.show()
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/98d119b7-216b-4a4c-928f-71ea05843fc5)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/e78d0e22-ff73-4a70-aa4a-d2e13e57f586)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/751dc807-9b98-44ff-9146-8ac7e2d8a952)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/bb40b821-d79c-434c-92c2-12e5d3ec8991)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/2a6d78f9-13c4-46b5-b2ab-67aa844e5e94)
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/36ec0272-489f-49f0-954e-3b5069de4fdf)

</details>

- These are some of the quick insights from this exercise:
  - Electronic check medium are the highest churners
  - Contract Type - Monthly customers are more likely to churn because of no contract terms, as they are free to go customers.
  - No Online security, No Tech Support category are high churners
  - Non senior Citizens are high churners

</details>

---

### 4️⃣ Fitting Model

<details><summary> Splitting Dataset  </summary> 
<br>
 
```python
X=df.drop('Churn',axis=1)
y=df['Churn']


# split dataset to test and training set (80% train, 20% test)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state = 1)
  
```
</details>
  
---  
###  5️⃣ Tuning

<br>
Firstly, I would write a function to evaluate the models (Confusion matrix & accuracy_score) and also applied it to Tunning Function too. 
</br>

<br>
<details><summary> Writing Evaluate Model Function  </summary>
  
 ```python

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve,classification_report
from sklearn.model_selection import cross_val_score

def EvaluateModel(model, y_test, y_pred, plot=False):
    
    #Confusion matrix
    # save confusion matrix and slice into four pieces
    confusion = metrics.confusion_matrix(y_true =y_test, y_pred = y_pred)
  

    # visualize Confusion Matrix
    sns.heatmap(confusion,annot=True,fmt="d") 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Training time end
    end_time = time.time()
    training_time = end_time - start_time

    #Metrics computed from a confusion matrix
    #Classification Accuracy: Overall, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('Classification Accuracy:', accuracy)
    
    #Classification Error: Overall, how often is the classifier incorrect?
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_pred))
    
    #Classification Report
    print('Classification Accuracy:' ,classification_report(y_test,y_pred))
    
  
    # Store the model's class name and its accuracy and training time in methodDict
    model_name = model.__class__.__name__
    methodDict[model_name] = {'accuracy': accuracy * 100, 'training_time': training_time}
 
 ```

</details>

<details><summary> Tunning Function </summary>
<br>

  - Because dataset is small, I still would like to use Random Search instead of Bayes, or gridsearch because I want to minimize the tuning time and better result,. In this case : I use RandomizedSearchCV

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 2)

def RandomSearch(model, param_dist):
  reg_bay = RandomizedSearchCV(estimator=model,
                    param_distributions=param_dist,
                    n_iter=20, 
                    cv=kf,
                    n_jobs=8,
                    scoring='accuracy',
                    random_state =3)
  reg_bay.fit(X_train,y_train)
  y_pred = reg_bay.predict(X_test)
  print('RandomSearch. Best Score: ', reg_bay.best_score_)
  print('RandomSearch. Best Params: ', reg_bay.best_params_)
  accuracy_score = EvaluateModel(model, y_test, y_pred, plot =True)

  ```
                                                                                      
</details>  


---  
### 6️⃣ Evaluate Models
  


<details><summary> Decision Tree </summary>

```python
model_2 = DecisionTreeClassifier()
param_dist = {
    'max_depth': [4, 6, 8, 10, 12],  
    'min_samples_leaf': [2, 4, 6, 8, 10], 
    'criterion': ['gini', 'entropy']  
}
print('Decision-Tree')
RandomSearch(model_2, param_dist)
    

      
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/f67f1fc5-3453-454a-a2ff-cbea398056d8)
<br>

  - As you can see that the accuracy is quite low, and as it's an imbalanced dataset, we shouldn't consider Accuracy as our metrics to measure the model, as Accuracy is cursed in imbalanced datasets.
Hence, we need to check recall, precision & f1 score for the minority class, and it's quite evident that the precision, recall & f1 score is too low for Class 1, i.e. churned customers.
Hence, moving ahead to call SMOTEENN (UpSampling + ENN)

</details>  

<details><summary> Decision Tree with SMOTEEN  </summary>

![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/35b4b96c-800b-4a4d-a150-5e57fedc9d2e)
<br>

  - Now we can see quite better results, i.e. Accuracy: 93 %, and a very good recall, precision & f1 score for minority class

</details>  

<details><summary> Random Forest </summary>

```python
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
param_dist = {'max_depth': list(range(1, 9)),
              "min_samples_leaf": list(range(1, 9)),
              "criterion": ["gini", "entropy"]}


print('Random Forest')

RandomSearch(model_rf, param_dist)

  
```
![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/611be907-b85a-432d-9195-b0c6e9243b2b)

</details>  

<details><summary>Random Forest with SMOTEEN </summary>

![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/54067250-97b9-4513-9512-a42cb7a00afb)
<br>

  - With RF Classifier, also we are able to get quite good results, infact better than Decision Tree. So we can use the Random Forest with SMOTEEN  best parameters
    
</details>  


---

### 7️⃣ Pickling the model

<br>

  - Our final model RF Classifier with SMOTEENN, is now ready and dumped in model.sav, which we will use and prepare API's so that we can access our model from UI.

![image](https://github.com/anhtuan0811/Telecom-Churn-Analysis/assets/143471832/880246cd-37be-4d12-94a0-3298b88e4d30)
  

</details>  

---

