#!/usr/bin/env python
# coding: utf-8

# Objective Association Rules Let's say you are a Machine Learning engineer working for a clothing company and you want to adopt new strategies to improve the company's profit.
# 
# Use this dataset and the association rules mining to find new marketing plans.
# 
# Note here that one of the strategies can be based on which items should be put together
# 
# dataset = [['Skirt', 'Sneakers', 'Scarf', 'Pants', 'Hat'],
# 
# ['Sunglasses', 'Skirt', 'Sneakers', 'Pants', 'Hat'],
# 
# ['Dress', 'Sandals', 'Scarf', 'Pants', 'Heels'],
# 
# ['Dress', 'Necklace', 'Earrings', 'Scarf', 'Hat', 'Heels', 'Hat'],
# ['Earrings', 'Skirt', 'Skirt', 'Scarf', 'Shirt', 'Pants']]
# 
# Bonus: try to do some visualization before applying the Apriori algorithm.
# 
# Let's do the same checkpoint but with a bigger dataset!

# In[203]:



import squarify
import warnings
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
plt.style.use('default')


# In[204]:


df = pd.read_csv('Market_Basket_Optimisation.csv')


# In[205]:


df


# In[206]:


df.head()


# In[207]:



df.info()


# In[208]:


df.shape


# In[209]:


df.describe()


# In[210]:


df.isnull().sum()


# # Data Visualizations

# In[211]:


# 1. Gather All Items of Each Transactions into Numpy Array
transaction = []
for i in range(0, data.shape[0]):
    for j in range(0, data.shape[1]):
        transaction.append(data.values[i,j])

transaction = np.array(transaction)

# 2. Transform Them a Pandas DataFrame
df = pd.DataFrame(transaction, columns=["items"]) 
df["incident_count"] = 1 # Put 1 to Each Item For Making Countable Table, to be able to perform Group By

# 3. Delete NaN Items from Dataset
indexNames = df[df['items'] == "nan" ].index
df.drop(indexNames , inplace=True)

# 4. Final Step: Make a New Appropriate Pandas DataFrame for Visualizations  
df_table = df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
# 5. Initial Visualizations
df_table.head(10).style.background_gradient(cmap='Blues')


# In[ ]:





# In[212]:


# Transform Every Transaction to Seperate List & Gather Them into Numpy Array
# By Doing So, We Will Be Able To Iterate Through Array of Transactions

transaction = []
for i in range(data.shape[0]):
    transaction.append([str(data.values[i,j]) for j in range(data.shape[1])])
    
transaction = np.array(transaction)

# Create a DataFrame In Order To Check Status of Top20 Items

top20 = df_table["items"].head(20).values
array = []
df_top20_multiple_record_check = pd.DataFrame(columns=top20)

for i in range(0, len(top20)):
    array = []
    for j in range(0,transaction.shape[0]):
        array.append(np.count_nonzero(transaction[j]==top20[i]))
        if len(array) == len(data):
            df_top20_multiple_record_check[top20[i]] = array
        else:
            continue
            

df_top20_multiple_record_check.head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[226]:


plt.figure(figsize=(14,8))
plt.title("FREQUENCY PLOT")
cnt = 45 # plot only first 'cnt' values
color = plt.cm.spring(np.linspace(0, 1, cnt))
df_sum.head(cnt).plot.bar(color = color)
plt.xticks(rotation = 'vertical')
plt.grid(False)
plt.axis('on')
plt.show()


# In[214]:


from mlxtend.preprocessing import TransactionEncoder

# Instantiate transaction encoder and fit in my list of sets data
encoder = TransactionEncoder().fit(transactions)

# Transform my actual data for a new representation
onehot = encoder.transform(transactions)

# Convert onehot encoded data to DataFrame
onehot = pd.DataFrame(onehot, columns = encoder.columns_)

onehot.iloc[:3]


# In[215]:



df.sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[216]:


# Transform Every Transaction to Seperate List & Gather Them into Numpy Array

transaction = []
for i in range(data.shape[0]):
    transaction.append([str(data.values[i,j]) for j in range(data.shape[1])])
    
transaction = np.array(transaction)
transaction


# In[217]:


te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)
dataset


# In[218]:


first50 = df_table["items"].head(50).values # Select Top50
dataset = dataset.loc[:,first50] # Extract Top50
dataset


# In[219]:


# Convert dataset into 1-0 encoding

def encode_units(x):
    if x == False:
        return 0 
    if x == True:
        return 1
    
dataset = dataset.applymap(encode_units)
dataset.head(10)


# In[ ]:





# In[220]:


# Extracting the most frequest itemsets via Mlxtend.
# The length column has been added to increase ease of filtering.

frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# In[221]:


frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.05) ]


# In[222]:


frequent_itemsets[ (frequent_itemsets['length'] == 3) ].head()


# In[223]:


# We can create our rules by defining metric and its threshold.

# For a start, 
#      We set our metric as "Lift" to define whether antecedents & consequents are dependent our not.
#      Treshold is selected as "1.2" since it is required to have lift scores above than 1 if there is dependency.

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("lift",ascending=False)


# In[ ]:


# Sort values based on confidence

rules.sort_values("confidence",ascending=False)


# In[ ]:


rules[~rules["consequents"].str.contains("mineral water", regex=False) & 
      ~rules["antecedents"].str.contains("mineral water", regex=False)].sort_values("confidence", ascending=False).head(10)


# In[ ]:


rules[rules["antecedents"].str.contains("ground beef", regex=False) & rules["antecedents_length"] == 1].sort_values("confidence", ascending=False).head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




