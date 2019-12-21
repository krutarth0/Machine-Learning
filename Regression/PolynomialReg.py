# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

#polynomial regression
polyreg = PolynomialFeatures(degree=4)
x_poly=polyreg.fit_transform(X)
polyreg.fit(x_poly,y)
regressor2=LinearRegression()
regressor2.fit(x_poly,y)

#adding more cureve to the graph
x_curve=np.arange(min(X),max(X),0.1)
x_curve=x_curve.reshape((len(x_curve),1))

#visualization
plt.scatter(X,y,color='red')
#plt.plot(X,regressor.predict(X),color='blue')
plt.plot(x_curve,regressor2.predict(polyreg.fit_transform(x_curve)),color='yellow')
plt.show()

