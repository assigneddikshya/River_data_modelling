#import packages for processing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import itertools
import statsmodels.api as sm

plt.style.use('fivethirtyeight')
mt.rcParams['axes.labelsize'] = 14
mt.rcParams['xtick.labelsize'] = 12
mt.rcParams['ytick.labelsize'] = 12
mt.rcParams['text.color'] = 'k'

#load data
data = pd.read_csv('river_data_exclusive.csv')
data = data[['TDS', 'pH',
       'Total_alkalinity', 'Total_hardness', 'Ca', 'Mg', 'Cl', 'SO4', 'NO3',
       'Fe', 'F', 'Total_coliform', 'fecal_coliform']]

#Visualisation of raw data
sns.pairplot(data, hue="TDS")

#import packages for model building and visualisation
import pandas as pd
from pandas_datareader import data as d
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from dtreeviz.trees import *
import graphviz

#Separating features and target
features = data.drop('TDS', axis=1) #X
feature_names = features.columns
target = data['TDS']  #y

#Visualisation of regression model decision tree using dtreeviz
from sklearn.tree import DecisionTreeRegressor
fig= plt.figure(figsize=(25,20))
regr = tree.DecisionTreeRegressor(max_depth=3)
regr.fit(features, target)
viz = dtreeviz(regr, features, target, target_name='TDS',
	      feature_names=features.columns, title='TDS regression',
	      colors={'title':'purple'}, scale=1.5)

viz

#regression model decision tree using DecisionTreeRegressor
#train and test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

rt = DecisionTreeRegressor(criterion='mse', max_depth=5)
model_r = rt.fit(X_train, y_train)
target_pred = model_r.predict(X_test)

#Evaluation
#MAE
print('Mean absolute error: ', metrics.mean_absolute_error(y_test, target_pred))
#MSE
print('Mean squared error: ', metrics.mean_squared_error(y_test, target_pred))
#RMSE
print('Root mean squared error: ', np.sqrt(metrics.mean_squared_error(y_test, target_pred)))
#R^2
print('R squared score: ', r2_score(y_test, target_pred))

'''
Mean absolute error:  4.423980186055856
Mean squared error:  33.25834962175575
Root mean squared error:  5.767005255915391
R squared score:  0.8091405922231601
'''
'''Conclusion R^2 is good, but MAE is very high.'''
