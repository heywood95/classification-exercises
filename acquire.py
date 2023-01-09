#!/usr/bin/env python
# coding: utf-8

# In[5]:


import seaborn as sns
import numpy as np
from pydataset import data

import os
import pandas as pd
import env
from env import get_connection


# In[6]:


def get_titanic_data():

    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        df_titanic = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

        
        df_titanic.to_csv(filename)

        
        return df_titanic  


# In[7]:


def get_iris_data():
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        df_iris = pd.read_sql('SELECT * FROM measurements JOIN species', get_connection('iris_db'))

        
        df_iris.to_csv(filename)

        
        return df_iris  


# In[8]:


def get_telco_data():
    filename = "telco.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        df_telco = pd.read_sql('SELECT * FROM customers JOIN contract_types JOIN internet_service_types JOIN payment_types', get_connection('telco_churn'))

        
        df_telco.to_csv(filename)

        
        return df_telco  


# In[ ]:




