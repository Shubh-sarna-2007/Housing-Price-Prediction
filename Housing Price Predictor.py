#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[ ]:


h_data = pd.read_csv("Housing.csv")


# In[ ]:


print(h_data.head())


# In[ ]:


h_data.columns


# In[ ]:


y = h_data['price']
y


# In[ ]:


x = h_data[['bedrooms','bathrooms','sqft_living']]
x


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state = 42) 


# In[ ]:


x_train


# In[ ]:


lr = LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


y_pred = lr.predict(x_test)


# In[ ]:


mean_absolute_error(y_test,y_pred)


# In[ ]:


houses = pd.DataFrame({'bedrooms':[5],'bathrooms':[4],'sqft_living':[3000]})
price = lr.predict(houses)


# In[ ]:


print(price)


# In[ ]:


plt.scatter(h_data['sqft_living'], h_data['price'])

plt.xlabel("sqft_living")
plt.ylabel("price")

plt.show()


# In[ ]:




