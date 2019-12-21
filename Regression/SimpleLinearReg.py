import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('./data/Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1:].values


#splitting the dataset into the traning set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


#fitting the linear regressoion to a model regressor
from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#pridicting
y_pred=regressor.predict(x_test)

#ploting
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.show()
