
# Polynomial regression model
# Use this if your dependent variable can be expressed as a sum of independent variable and then verify the accuracy of the model.


# 1) importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2) import dataset

dataset = pd.read_csv('Dataset_file.csv') # Replace the dataset file name here.
x = dataset.iloc[:, 1:-1].values # change if necessary, the dependent and independent column variable index here
y = dataset.iloc[:, -1].values


# Training the polynomial regression model on dataset

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_regressor= LinearRegression()
lin_regressor.fit(x_poly,y)


# 4) visualisation of polynomial regression model results

plt.scatter(x,y,color='red')
plt.plot(x,lin_regressor.predict(x_poly),color='blue')
plt.title('(polynomial regression)')
plt.xlabel('X_label') # Replace the plot labels here
plt.ylabel('Y_label')
plt.show()

# 5) predict new reading"

lin_regressor.predict(poly_reg.fit_transform([[num]])) # change 'num' here to any desired numerical value.