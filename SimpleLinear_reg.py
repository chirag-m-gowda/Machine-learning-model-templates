# Simple linear regression model:
""" 
This model provides the best results when used with a dataset having 2 parameters 
which have linear mathematical relationship between them. Then this model can be used to
draw a regression line and predict the values of the output parameter for any given input.
"""
# 1) Import the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2)import dataset

dataset= pd.read_csv('YOUR_DATASET.csv') # replace your dataset file name in this line
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(x)
print(y)

# 3) Split dataset into train and test (80/20 split)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) #change here for varying split ratio.

# 4) training simple linear regression model on training dataset

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)

# 5) Predicting the test set results

y_pred=regressor.predict(x_test)

# 6) Visualising the training set results

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title(" abc (training set)") # modify the plot 'title' and 'x','y' labels.
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 7) Visualising the test set results 

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title("abc (test set)")  # modify the plot 'title' and 'x','y' labels.
plt.xlabel("x")
plt.ylabel("y")
plt.show()