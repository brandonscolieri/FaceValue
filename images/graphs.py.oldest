#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


col_names = ['epochs', 'time', 'loss', 'acc', 'val_loss', 'val_acc']
df_worst = pd.read_csv('worst_model.csv', delimiter=',', names=col_names)
df_second_best = pd.read_csv('second_best_model.csv', delimiter=',', names=col_names)
df_best = pd.read_csv('best_models_points_to_plot.csv', delimiter=',', names=col_names)

# Remove columns that are useless
worst = df_worst.drop(df_worst.index[0])
second_worst = df_second_best.drop(df_second_best.index[0])
best = df_best.drop([df_best.index[0], df_best.index[-1]])


# In[29]:


# Helper function to change strings in columns to numeric datatypes

def _numeric_helper(dataframe, cols_to_change):
    for i in cols_to_change:
        dataframe[i] = pd.to_numeric(dataframe[i])
    return dataframe


# In[30]:


# Cast to proper types
datatype_dict = {'epochs': np.uint8, 'loss': np.float64, 'acc': np.float64, 'val_loss': np.float64, 'val_acc': np.float64}

df_w = _numeric_helper(worst, datatype_dict)
df_2w = _numeric_helper(second_worst, datatype_dict)
df_b = _numeric_helper(best, datatype_dict)


# In[3]:


# # Helper function

# def _helper(dataframe, cols):
#     return (dataframe.loc[:, col].to_numpy() for col in cols)


# ## Plots for worst model

# In[52]:


# epochs_worst, _, loss_worst, acc_worst, val_loss_worst, val_acc_worst = _helper(worst, col_names)
# plt.axes()
# plt.yticks(np.arange(0, 4, step=0.2))
# plt.plot(epochs_worst, loss_worst, label='training')
# plt.plot(epochs_worst, val_loss_worst, label='validation')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('Loss of worst model')
# plt.legend()
# plt.show()
sns.lineplot(x='epochs', y='loss', data=df_w)
plt.title('Training loss for worst model')
plt.show()
sns.lineplot(x='epochs', y='val_loss', data=df_w)
plt.title('Validation loss for worst model')
plt.show()


# In[53]:


# plt.plot(epochs_worst, acc_worst, label='training')
# plt.plot(epochs_worst, val_acc_worst, label='validation')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.title('Accuracies of worst model')
# plt.legend()
# plt.show()

gr = sns.lineplot(x='epochs', y='acc', data=df_w)
plt.title('Training accuracy for worst model')
plt.show()
gr = sns.lineplot(x='epochs', y='val_acc', data=df_w)
plt.title('Validation accuracy for worst model')
plt.show()


# ## Plots for second-worst model

# In[55]:


sns.lineplot(x='epochs', y='loss', data=df_2w)
plt.title('Training loss for second worst model')
plt.show()
sns.lineplot(x='epochs', y='val_loss', data=df_2w)
plt.title('Validation loss for second worst model')
plt.show()


# In[56]:


sns.lineplot(x='epochs', y='acc', data=df_2w)
plt.title('Training accuracy for second worst model')
plt.show()
sns.lineplot(x='epochs', y='val_acc', data=df_2w)
plt.title('Validation accuracy for second worst model')
plt.show()


#  ## Plots for best model

# In[59]:


sns.lineplot(x='epochs', y='loss', data=df_b)
plt.title('Training loss for best model')
plt.show()
sns.lineplot(x='epochs', y='val_loss', data=df_b)
plt.title('Validation loss for best model')
plt.show()


# In[62]:


sns.lineplot(x='epochs', y='acc', data=df_b)
plt.title('Training accuracy for best model')
plt.show()
sns.lineplot(x='epochs', y='val_acc', data=df_b)
plt.title('Validation accuracy for best model')
plt.show()


# In[ ]:




