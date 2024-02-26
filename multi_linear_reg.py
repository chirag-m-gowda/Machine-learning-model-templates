# Multiple linear regression model
# This model is to be used when there are multiple independent variables and one dependent variable.

"""
Check the accuracy of this model to validate if there is linear relationship among the variables,
if not providing satisfactory accuracy, try other models. 
"""

# Importing the libraries:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset:

dataset=pd.read_csv('Data_set_name.csv') # Change the file name of dataset here.
x=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,-1].values # Last row contains dependent variables.

# Hotencoding categorical data ( if categorical data present in dependent variables, i.e values in the column, other than numbers use this block, else comment this out.)

# One hot encoding of independent variable columns:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[Column_index])],remainder='passthrough') # Replace the categorical data column index number here. 
x = np.array(ct.fit_transform(x))

# Splitting the dataset into training and test set (80/20 split):

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Train the model on the training set:

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predict the test set results:

y_pred= regressor.predict(x_test)
np.set_printoptions(precision=2) # adjust precision of floating point numbers here.
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))