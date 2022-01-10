import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('./0_Data_Generation/data/linear_data.csv')
X = dataset.iloc[ : , : 1 ].values
Y = dataset.iloc[ : , 1 ].values

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 0) 

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

plt.figure("linear_regression")
ax1 = plt.subplot(121)
ax1.axis([0,10,0,10])
ax1.scatter(X_train , Y_train, color = 'red')
ax1.plot(X_train , regressor.predict(X_train), color ='blue')
ax2 = plt.subplot(122)
ax2.axis([0,10,0,10])
ax2.scatter(X_test , Y_test, color = 'red')
ax2.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()