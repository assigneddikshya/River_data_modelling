#1. Averaging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score

import os
os.chdir("C:\\Users\\diksh\\OneDrive\\Desktop\\Independent projects\\Oretes")

data = pd.read_csv('river_data_exclusive.csv')
data = data[['TDS', 'pH',
       'Total_alkalinity', 'Total_hardness', 'Ca', 'Mg', 'Cl', 'SO4', 'NO3',
       'Fe', 'F', 'Total_coliform', 'fecal_coliform']]
train = data.drop('TDS', axis=1) #X
target = data['TDS'] #y
X_train, X_test,y_train, y_test = train_test_split(train, target, test_size=0.2)

model_1 = LinearRegression()
model_2 = xgb.XGBRegressor()
model_3 = RandomForestRegressor()

m1 = model_1.fit(X_train, y_train)
m2 = model_2.fit(X_train, y_train)
m3 = model_3.fit(X_train, y_train)

pred_1 = m1.predict(X_test)
pred_2 = m2.predict(X_test)
pred_3 = m3.predict(X_test)

pred_final = (pred_1+pred_2+pred_3)/3.0

print(mean_squared_error(y_test,pred_final)) #24.28
#R^2
print('R squared score: ', r2_score(y_test,pred_final)) #0.86

print('Mean absolute error: ', metrics.mean_absolute_error(y_test, pred_final)) #3.84

''' A bit less, R^2 a bit better. '''


