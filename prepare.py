#!/usr/bin/env python
# coding: utf-8

# In[3]:


import env
import pandas as pd
import os
import seaborn as sns
import numpy as np
from pydataset import data
from env import get_connection

import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquire


# In[4]:


iris = acquire.get_iris_data()


# In[5]:


cols_to_drop = ['Unnamed: 0', 'measurement_id', 'species_id', 'species_id.1']
iris = iris.drop(columns=cols_to_drop)


# In[6]:


iris.rename(columns = {'species_name': 'species'}, inplace=True)


# In[7]:


dummy_iris = pd.get_dummies(iris.species, drop_first=True)
iris = pd.concat([iris, dummy_iris], axis=1)


# In[8]:


def prep_iris(iris):
    
    cols_to_drop = ['Unnamed: 0', 'measurement_id', 'species_id', 'species_id.1']
    iris = iris.drop(columns=cols_to_drop)
    iris.rename(columns = {'species_name': 'species'}, inplace=True)
    dummy_iris = pd.get_dummies(iris.species, drop_first=True)
    iris = pd.concat([iris, dummy_iris], axis=1)
    
    return iris


# In[9]:


titanic = acquire.get_titanic_data()


# In[10]:


cols_to_drop = ['deck', 'embarked', 'class', 'age']
titanic = titanic.drop(columns=cols_to_drop)


# In[11]:


dummy_titanic = pd.get_dummies(titanic[['sex','embark_town']], dummy_na=False, drop_first=[True, True])
titanic = pd.concat([titanic, dummy_titanic], axis=1)


# In[12]:


def prep_titanic(df):
    
    cols_to_drop = ['deck', 'embarked', 'class', 'age']
    titanic = titanic.drop(columns=cols_to_drop)
    dummy_titanic = pd.get_dummies(titanic[['sex','embark_town']], dummy_na=False, drop_first=[True, True])
    titanic = pd.concat([titanic, dummy_titanic], axis=1)
    titanic.dropna()
    return titanic


# In[13]:


telco = acquire.get_telco_data()


# In[14]:


cols_to_drop = ['contract_type_id', 'internet_service_type_id', 'payment_type_id']
telco = telco.drop(columns=cols_to_drop)


# In[15]:


dummy_telco = pd.get_dummies(telco[['gender','partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn']], dummy_na=False, drop_first=[True, True])
telco = pd.concat([telco, dummy_telco], axis=1)


# In[16]:


def prep_telco(telco):
    
    cols_to_drop = ['contract_type_id', 'internet_service_type_id', 'payment_type_id']
    telco = telco.drop(columns=cols_to_drop)
    dummy_telco = pd.get_dummies(telco[['gender','partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn']], dummy_na=False, drop_first=[True, True])
    telco = pd.concat([telco, dummy_telco], axis=1)
    return telco


# In[17]:


train, test = train_test_split(telco, 
                               train_size = 0.8,
                               random_state=42,
                              stratify=telco.churn)


# In[18]:


train, val = train_test_split(train,
                             train_size = 0.7,
                             random_state=42,
                             stratify=train.churn)


# In[19]:


def split_data(df, target=''):
    
    train, test = train_test_split(df, 
                               train_size = 0.8,
                               random_state=1349,
                              stratify=df[target])
    train, val = train_test_split(train,
                             train_size = 0.7,
                             random_state=1349,
                             stratify=train[target])
    return train, val, test


# In[20]:


iris = acquire.get_iris_data()


# In[21]:


train, validate, test = split_data(iris, target='species_name')


# In[22]:


train.size, validate.size, test.size


# In[23]:


titanic_train, titanic_val, titanic_test = split_data(
titanic, target='survived')


# In[24]:


titanic_train.size, titanic_val.size, titanic_test.size


# In[25]:


telco_train, telco_val, telco_test = split_data(
telco, target='churn')


# In[26]:


telco_train.size, telco_val.size, telco_test.size


# In[ ]:




