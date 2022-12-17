#!/usr/bin/env python
# coding: utf-8

# In[171]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[172]:


df=pd.read_csv(r'C:\Users\paas9002\Desktop\perrin-freres-monthly-champagne-.csv')


# In[173]:



df.head()


# In[174]:


df.tail()


# In[175]:


df.columns=["Month","Sales"]
df.head()


# In[176]:


df.drop(106,axis=0,inplace=True)


# In[177]:


df.tail()


# In[178]:


df.drop(105,axis=0,inplace=True)


# In[179]:


df.tail()


# In[180]:


df['Month']=pd.to_datetime(df['Month'])


# In[181]:


df.head()


# In[182]:


df.set_index('Month',inplace=True)


# In[183]:



df.head()


# In[184]:


df.describe()


# In[185]:


df.plot()


# In[186]:


from statsmodels.tsa.stattools import adfuller


# In[187]:


test_result=adfuller(df['Sales'])


# In[188]:


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[189]:



adfuller_test(df['Sales'])


# In[190]:


df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)


# In[191]:


df.head(14)


# In[192]:


adfuller_test(df['Seasonal First Difference'].dropna())


# In[193]:


df['Seasonal First Difference'].plot()


# In[194]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()


# In[195]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm


# In[196]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)


# In[197]:


from statsmodels.tsa.arima_model import ARIMA


# In[198]:


train =df.iloc[:-21]
test=df.iloc[-21:]
print(train.shape,test.shape)


# In[199]:


model=ARIMA(train['Sales'],order=(1,1,1))
model_fit=model.fit()


# In[200]:


model_fit.summary()


# In[201]:


df['forecast']=model_fit.predict(start=84,end=104,dynamic=True)
df['forecast'].plot(legend=True)
train['Sales'].plot(legend=True)


# In[202]:


model=sm.tsa.statespace.SARIMAX(train['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()


# In[203]:


df['forecast']=results.predict(start=84,end=104,dynamic=True)
df['forecast'].plot(legend=True)
test['Sales'].plot(legend=True)


# In[204]:


test['Sales'].mean()


# In[205]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(df['forecast'][84:],test['Sales']))
print(rmse)


# In[206]:


from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]


# In[207]:



future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)


# In[208]:


future_datest_df.tail()


# In[209]:



future_df=pd.concat([df,future_datest_df])


# In[214]:


future_df['forecast'] = results.predict(start = 104, end = 128, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8)) 


# In[215]:


future_df.iloc[104:]


# In[ ]:





# In[ ]:




