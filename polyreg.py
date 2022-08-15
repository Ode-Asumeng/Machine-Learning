#polynomial regresion

#importing libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
 
#load/read dataset
dataset = pd.read_csv('position_salaries.csv')

#separating dependent variables
X = dataset.iloc[:,1:2].values #independent variable is level. It should always be a matrix
Y = dataset.iloc[:,2].values #dependent variable is salary. It should always be a vector

#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2) #transforming tool
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, Y)  #Polynomial model ready

#visualizing the linear regression prediction result
plt.scatter(X,Y)
plt.plot(X, lin_reg.predict(X), color = 'red')
plt.title('Salary Prediction with Linear Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Predicted Salary')
plt.show()


#visualizing the linear regression prediction result
plt.scatter(X,Y)
plt.plot(X, poly_reg.predict(poly.fit_transform(X)), color = 'green')
plt.title('Salary Prediction with Polynomial Regression Model')
plt.xlabel('Position Level')
plt.ylabel('Predicted Salary')
plt.show()

#optimizing our polynomial regression model by tuning our parameters
poly = PolynomialFeatures(degree=4) #transforming tool
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, Y)

#predicting new result with linear regression
lin_reg.predict([[11]])

#predicting new result with multiple regression
poly_reg.predict(poly.fit_transform([[11]]))
