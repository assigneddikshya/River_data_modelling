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

data = pd.read_csv('river_data_exclusive.csv')
data = data[['Date_of_collection', 'TDS', 'pH',
       'Total_alkalinity', 'Total_hardness', 'Ca', 'Mg', 'Cl', 'SO4', 'NO3',
       'Fe', 'F', 'Total_coliform', 'fecal_coliform']]

#check for a trend
# trend by year
m= data.groupby('Date_of_collection')['TDS'].mean().reset_index()
m['Date_of_collection'] = pd.to_datetime(m['Date_of_collection'],format='%Y-%m-%d')
m.set_index('Date_of_collection', inplace=True)
m.plot()
plt.show()

'''Hence, trend exists.'''

#check for stationarity
from pandas import Series
from statsmodels.tsa.stattools import adfuller

result = adfuller(m)

print('ADF statistics: %f' % result[0])
print('p-value %f' % result[1]) 
print('Critical value: ')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

'''
ADF statistics: -0.550328
p-value 0.881816
Critical value: 
	1%: -3.575
	5%: -2.924
	10%: -2.600

Hence, p-value > 0.05, data is unstationary.'''

#make data stationary
#diff
data['TDS'] = data['TDS'].diff().fillna(data['TDS'].mean())

m= data.groupby('Date_of_collection')['TDS'].mean().reset_index() #can use sum too
m['Date_of_collection'] = pd.to_datetime(m['Date_of_collection'],format='%Y-%m-%d')
m.set_index('Date_of_collection', inplace=True)
m.plot()
plt.show() 

y = m.resample('M').sum()
y.loc[y['TDS'] == '0.00', 'TDS'] = y.mean()
y = y.fillna(y.mean())

from pandas import Series
from statsmodels.tsa.stattools import adfuller

result = adfuller(y)

print('ADF statistics: %f' % result[0])
print('p-value %f' % result[1]) #0.000< 0.05, hence, stationary
print('Critical value: ')
for key, value in result[4].items():
       print('\t%s: %.3f' % (key, value))

'''
ADF statistics: -8.446071
p-value 0.000000
Critical value: 
	1%: -3.585
	5%: -2.928
	10%: -2.602

p-value < 0.05, hence data is stationary.'''

#Decompostion
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(y, period=5)
plt.plot(y, label='Original')
plt.show()

trend = decomposition.trend
plt.plot(trend, label='trend')
plt.legend(loc='best')
plt.show()

seasonal = decomposition.seasonal
plt.plot(seasonal, label='seasonal')
plt.legend(loc='best')
plt.show()

residual = decomposition.resid
plt.plot(residual, label='residual')
plt.legend(loc='best')
plt.show()

#SARIMA model
p=d=q=range(0,2) #p,d,q to be selected in the range of 0 to 2 -> values can be 0 or 1
pdq = list(itertools.product(p,d,q))
seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]

#Find model with the least AIC value
from pylab import rcParams
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order = param,
                                            seasonal_order =param_seasonal,
                                            enforce_stationarity= False,
                                            enforce_invertibility= False)
            results = mod.fit()
            print('ARIMA{} x {} - AIC: {}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                   order=(1,0,0),
                                   seasonal_order=(1,1,0,12),
                                   enforce_stationarity = False,
                                   enforce_invertibility = False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16,8))
plt.show()

#Prediction
pred = results.get_prediction(start=pd.to_datetime('2019-05-31'),dynamic = False)
pred_ci = pred.conf_int()  #to build the confidence interval
ax = y['2015-05-12':].plot(label='observed') #actual observed data
pred.predicted_mean.plot(ax=ax, label='One step ahead forecast', alpha=1, figsize=(14,7)) 
#plotting the observed data from 2019 along with confidence interval and predicted value of data from 2019
ax.fill_between(pred_ci.index,
                  pred_ci.iloc[:,0],#lower bound of CI
                  pred_ci.iloc[:,1],#upper bound of CI
                  color='k',
                  alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('sales')
plt.legend()
plt.show()

#Forecast
pred_uc = results.get_forecast(steps=13) #will give you the prediction for 13 months in future(2018)-> 12 months of 2018 & 1st month of 2019
pred_ci = pred_uc.conf_int()
ax = y.plot(label = 'observed', figsize=(14,7))
pred_uc.predicted_mean.plot(ax=ax, label='forecast')
ax.fill_between(pred_ci.index,
                   pred_ci.iloc[:,0],
                   pred_ci.iloc[:,1],
                   color='k',
                   alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('sales')
print(pred_ci)
plt.legend()
plt.show()

