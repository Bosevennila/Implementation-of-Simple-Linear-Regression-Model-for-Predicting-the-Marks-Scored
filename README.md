# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: CHANDRAPRIYADHASHINI C
RegisterNumber:  212223240019

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/36285a85-d8ab-48c6-b99c-2e85be00449f)
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/9b2188e6-5c7c-4e6c-8073-1572e29b4b66)
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/1c5979b2-2a1c-499f-b221-b96e855dc5b5)
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/b45251d9-c926-4a5f-a678-0bd7b3a167d6)
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/1f8edc88-78a3-4344-9ec2-6d65de8ed86e)
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/2e3ea699-ca7c-4e75-860a-85e861b9c105)
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/d671176c-0720-4940-8b29-d674d138f4d2)

![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/99401d94-d4c9-4af5-9d23-09693845727d)
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/257dc75c-7ce1-450e-9c2e-19015d62d75a)
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/b2196900-59a0-41f6-b4d0-d6945a62482b)
![image](https://github.com/Bosevennila/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144870486/c5efb592-e0c6-4102-949f-0f9cc953fdbf)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
