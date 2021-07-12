#!/usr/bin/env python
# coding: utf-8

# # Experimentation and uplift testing

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv("QVI_data.csv")


# In[2]:


data.info()


# In[3]:


data.head()

## Select control stores
-The client has selected store numbers 77, 86 and 88 as trial stores 
-Want control stores to be established stores that are operational for the entire observation period.
We would want to match trial stores to control stores that are similar to the trial
store prior to the trial period of Feb 2019 in terms of :
- Monthly overall sales revenue
- Monthly number of customers
- Monthly number of transactions per customer
# ### Let's first create the metrics of interest and filter to stores that are present throughout the pre-trial period.

# In[4]:


#### Calculate these measures over time for each store 
#### Over to you! Add a new month ID column in the data with the format yyyymm.

data['MONTH_ID'] = [''.join(x.split('-')[0:2]) for x in data.DATE]
data['MONTH_ID'] = pd.to_numeric(data['MONTH_ID'])
data.MONTH_ID.head()
## Check format of the serie
data.MONTH_ID.dtype


# In[5]:


# Next, we define the measure calculations to use during the analysis.
# Over to you! For each store and month calculate:
# 1. Total sales
# 2. Number of customers
# 3. Transactions per customer
# 4. Chips per customer
# 5. Average price per unit.


# In[6]:


print(data.STORE_NBR.nunique())
print(data.MONTH_ID.nunique())


# In[7]:


# 1. Total Sales
tot_sales = data.groupby(['STORE_NBR', 'MONTH_ID']).TOT_SALES.sum()


# In[8]:


# 2. Number of customers
n_customers = data.groupby(['STORE_NBR', 'MONTH_ID']).LYLTY_CARD_NBR.nunique()


# In[9]:


# 3. Average transactions per customer
txt_percustomer = data.groupby(['STORE_NBR', 'MONTH_ID']).TXN_ID.nunique() / n_customers


# In[10]:


# 4. Chips per customer
total_quantity = data.groupby(['STORE_NBR', 'MONTH_ID']).PROD_QTY.sum()
chips_percustomer = total_quantity / n_customers


# In[11]:


# 5. Average price per unit
total_sales = data.groupby(['STORE_NBR', 'MONTH_ID']).TOT_SALES.sum()
avg_price_unit = total_sales/total_quantity


# In[12]:


metrics = pd.concat([tot_sales, n_customers, txt_percustomer, chips_percustomer, avg_price_unit], axis = 1)
metrics.columns = ['Total_Sales', 'N_Customers', 'Transaction_PC', 'Chips_PC', 'Avg_price_chips']
metrics = metrics.reset_index()
metrics


# In[13]:


#### Filter to the pre-trial period and stores with full observation periods

# Stores with full observations periods 
counts = metrics['STORE_NBR'].value_counts()
index = counts[counts==12].index
stores_full_obs = metrics[metrics['STORE_NBR'].isin(index)]
stores_full_obs

#Filter to the pre-trial period (201807-201901)
stores_full_obs_pre_trial = stores_full_obs[stores_full_obs['MONTH_ID'] < 201902]
stores_full_obs_pre_trial


# ## Now we need to work out a way of ranking how similar each potential control store is to the trial store. 
# We can calculate how correlated the performance of each store is to the trial store. (*Let's write a function for this so that we don't have to calculate this for each trial store and control store pair.*)
# 
# 

# ###  Over to you! Create a function to calculate correlation for a measure, looping through each control store.
# 
# #### Let's define:
#  1. **InputTable**: metric table with potential comparison stores.
#  2. **MetricCol**: store metric used to calculate correlation on.
#  3. **StoreComparison**: the store number of the trial store (trial store).

# In[14]:


def calculateCorrelation(inputTable, metricCol, storeComparison):
    output = pd.DataFrame({'Store1': [], 'Store2': [], 'Correlation': []})
    trial = inputTable.loc[inputTable['STORE_NBR'] == storeComparison, metricCol]
    trial.reset_index(drop = True, inplace = True)
    storeNumbers = inputTable['STORE_NBR'].unique()
    for i in storeNumbers:
        control = inputTable.loc[inputTable['STORE_NBR'] == i, metricCol]
        control.reset_index(drop = True, inplace = True)
        output = output.append({'Store1': storeComparison, 'Store2': i, 'Correlation': control.corr(trial)}, ignore_index = True)
    return output


# ### Apart from correlation, we can also calculate a standardised metric based on the absolute difference between the trial store's performance and each control store's performance.
# 

# In[15]:


#### Standardise the magnitude distance so that the measure ranges from 0 to 1

measure = trial_store - control_store
minDist = min(measure)
maxDist = max(measure)

magnitudeMeasure := 1 - (measure - minDist)/(maxDist - minDist)]
# In[16]:


#### Create a function to calculate a standardised magnitude distance for a measure, looping through each control store
import numpy as np
def calculateMagnitudeDistance(inputTable, metricCol, storeComparison):
    output = pd.DataFrame({'Store1': [], 'Store2': [], 'MagnitudeMeasure': []})
    trial = inputTable.loc[inputTable['STORE_NBR'] == storeComparison, metricCol]
    trial.reset_index(drop = True, inplace = True)
    storeNumbers = inputTable['STORE_NBR'].unique()
    for i in storeNumbers:
        control = inputTable.loc[inputTable['STORE_NBR'] == i, metricCol]
        control.reset_index(drop = True, inplace = True)
        diff = abs(trial - control)
        s_diff = np.mean(1-((diff-min(diff))/(max(diff)-min(diff))))
        output = output.append({'Store1': storeComparison, 'Store2': i, 'MagnitudeMeasure': s_diff}, ignore_index = True)
    return output


# ## Now let's use the functions to find the control stores!

# ##### We'll select control stores based on:
# 
# -How similar are to the trial stores in terms of:
# 1. Monthly total sales in dollar amounts
# 2. Monthly number of customers 
# 
# We will need to use our functions to get:
# -Four scores, two for each of total sales and total customers.

# ## Trial Store 77

# In[17]:


storeComparison = 77
# Calculate correlations against store 77 using total sales and number of customers
corr_sales = calculateCorrelation(stores_full_obs_pre_trial, 'Total_Sales', storeComparison)
corr_ncustomers = calculateCorrelation(stores_full_obs_pre_trial, 'N_Customers', storeComparison)

# Calculate magnitude measure against store 77 using total sales and number of customers
magnitude_sales = calculateMagnitudeDistance(stores_full_obs_pre_trial, 'Total_Sales', storeComparison)
magnitude_ncustomers = calculateMagnitudeDistance(stores_full_obs_pre_trial, 'N_Customers', storeComparison)


# In[18]:


print(corr_sales.head())
print(corr_ncustomers.head())
print(magnitude_sales.head())
print(magnitude_ncustomers.head())


# ## We'll need to combine the all the scores calculated using our function to create a composite score to rank on. 
# 
# Let's take a simple average of the correlation and magnitude scores for each driver. 
# 
# Note that:
# 
# 1. If we consider it more important for the trend of the drivers to be similar: 
# 
# We can **increase** the weight of the correlation score *(a simple average gives a weight of 0.5 to the corr_weight).
# 
# 2. If we consider the absolute size of the drivers to be more important:
# 
# We can **lower** the weight of the correlation score.

# In[19]:


#### Over to you! Create a combined score composed of correlation and magnitude, by first merging the correlations table with the magnitude table.
corr_sales = corr_sales.rename(columns={'Correlation':'Correlation_Sales'})
magnitude_sales = magnitude_sales.rename(columns={'MagnitudeMeasure': 'Magnitude_Sales'})


# In[20]:


corr_ncustomers = corr_ncustomers.rename(columns={'Correlation':'Correlation_NCustomers'})
magnitude_ncustomers = magnitude_ncustomers.rename(columns={'MagnitudeMeasure':'Magnitude_Ncustomers'})


# In[21]:


sales = pd.merge(corr_sales, magnitude_sales, left_index=False, right_index=False)
ncustomers = pd.merge(corr_ncustomers, magnitude_ncustomers, left_index=False, right_index=False)


# In[22]:


#### Hint: A simple average on the scores would be 0.5 * corr_measure + 0.5 * mag_measure
corr_measure = 0.5
# 1. Sales Table
sales['combined_score_sales'] = ((corr_measure * sales['Correlation_Sales']) + ((1-corr_measure)*sales['Magnitude_Sales']))
sales.head()


# In[23]:


# 2. N_Customers Table
ncustomers['combined_score_ncustomers'] = ((corr_measure * ncustomers['Correlation_NCustomers']) + ((1-corr_measure)*ncustomers['Magnitude_Ncustomers']))
ncustomers.head()


# In[24]:


#Now we have a score for each of total number of sales and number of customers. Let's combine the two via a simple average.
### Over to you! Combine scores across the drivers by first merging our sales scores and customer scores into a single table.

final_score = pd.merge(sales, ncustomers, left_index=False, right_index=False)
final_score = final_score.drop(['Correlation_Sales','Magnitude_Sales', 'Correlation_NCustomers', 'Magnitude_Ncustomers'], axis=1)
final_score['final_score'] = ((corr_measure * final_score['combined_score_sales']) + ((1-corr_measure)*final_score['combined_score_ncustomers']))
final_score.head()

The store with the highest score is then selected as the control store since it is most similar to the trial store.
#  
# Select control stores based on the highest matching store (closest to 1 but  not the store itself, i.e. the second ranked highest store)

# In[25]:


#### Over to you! Select the most appropriate control store for trial store 77 by finding the store with the highest final score
final_score.sort_values('final_score', ascending=False).head(10)


# #### Now that we have found a control store, let's check visually if the drivers are indeed similar in the period before the trial.

# In[26]:


#### Visual checks on trends based on the drivers
#We'll look at total sales first.


# In[27]:


stores = [77, 233]
filtered_metrics = stores_full_obs_pre_trial[stores_full_obs_pre_trial.STORE_NBR.isin(stores)]
filtered_metrics.STORE_NBR.unique()


# In[28]:


filtered_metrics.head()


# In[29]:


filtered_metrics['MONTH_ID'] = pd.to_datetime(filtered_metrics['MONTH_ID'], format="%Y%m")
filtered_metrics['MONTH_ID'].head()


# In[192]:


trial = [77]
trial_store = filtered_metrics[filtered_metrics.STORE_NBR.isin(trial)]
c = pd.pivot_table(trial_store, values='Total_Sales', index='MONTH_ID')

control=[233]
control_store = filtered_metrics[filtered_metrics.STORE_NBR.isin(control)]
d = pd.pivot_table(control_store, values='Total_Sales', index='MONTH_ID')

print(c.head())
print(d.head())


# In[193]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(c, '-o', d, '-o')


# In[194]:


plt.xlabel('Period')
plt.ylabel('Dollar Amounts')
plt.title('Monthly Total Sales Dollar Amounts')
plt.legend(['Trial Store (77)', 'Control Store (233)'])


# In[33]:


#### Next, number of customers.
#### Over to you! Conduct visual checks on customer count trends by comparing the trial store to the control 
#### store and other stores.


# In[34]:


trial_store0 = 77
control_store0 = 233


# In[35]:


# Create a new dataframe 'pastSales'
measures = stores_full_obs_pre_trial

# Create a new column within 'pastSales' which categorises store type
store_type = []

for i in measures['STORE_NBR']:
    if i == trial_store0:
        store_type.append('Trial Store')
    elif i == control_store0:
        store_type.append('Control Store')
    else:
        store_type.append('Other Stores')

measures['store_type'] = store_type
measures['MONTH_ID'] = pd.to_datetime(measures['MONTH_ID'], format="%Y%m")
measures.head()


# In[195]:


trial = [77]
trial_store = measures[measures.STORE_NBR.isin(trial)]
c = pd.pivot_table(trial_store, values='N_Customers', index='MONTH_ID')

control=[233]
control_store = measures[measures.STORE_NBR.isin(control)]
d = pd.pivot_table(control_store, values='N_Customers', index='MONTH_ID')


# In[196]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(c, '-o', d, '-o')


# In[206]:


plt.xlabel('Period')
plt.ylabel('Number of Customers')
plt.title('Monthly number of customers')
plt.legend(['Trial Store (77)', 'Control Store (233)', 'Other Stores'])


# #### The trial period goes from the start of February 2019 to April 2019. We now want to see if there has been an uplift in overall chip sales. 

#  We'll start with scaling the control store's sales to a level similar to trial stores for any differences between 
#  the two stores outside of the trial period. 

# In[40]:


stores_full_obs[stores_full_obs['STORE_NBR'] == 77]['Total_Sales']


# In[41]:


stores_full_obs[stores_full_obs['STORE_NBR'] == 77]['Total_Sales'].sum()


# In[42]:


pre_trial_trial = stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 77]['Total_Sales'].sum()
pre_trial_control = stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 233]['Total_Sales'].sum()


# In[43]:


# Scale pre-trial control sales to match pre-trial trial store sales
scalingfactor = pre_trial_trial/pre_trial_control
scalingfactor


# In[44]:


#### Apply the scaling factor
stores_full_obs_233 = stores_full_obs[stores_full_obs['STORE_NBR'] == 233]
stores_full_obs_233['scaled_control_sales'] = stores_full_obs_233['Total_Sales']*scalingfactor
stores_full_obs_233.reset_index(drop=True).head()


# In[45]:


### Now that we have comparable sales figures for the control store, we can calculate the percentage difference 
### between the scaled control sales and the trial store's sales during the trial period.

scaled_control =stores_full_obs_233[['MONTH_ID', 'scaled_control_sales']].reset_index(drop=True)

trialsales = metrics[metrics['STORE_NBR'] == 77].reset_index(drop=True)
trialsales = trialsales[['MONTH_ID', 'Total_Sales']]

#### Over to you! Calculate the percentage difference between scaled control sales and trial sales
per_diff = pd.merge(scaled_control, trialsales, left_index=False, right_index=False)
per_diff['percentagedifference'] = abs(per_diff.scaled_control_sales - per_diff.Total_Sales) / per_diff.scaled_control_sales
per_diff


# # Let's see if the difference is significant!
# ### Null Hypothesis: Trial Period is the same as Pre-Trial Period

# In[46]:


## Let's take the standard deviation based on the scaled percentage difference in the pre-trial period:
std = (per_diff[per_diff['MONTH_ID'] < 201902]['percentagedifference']).std()
std


# In[47]:


# Note that there are 8 months in the pre-trial period hence 8 - 1 = 7 degrees of freedom
df = 7


#  We will test with a null hypothesis of there being 0 difference between trial and control stores.
# ### Null Hypothesis: There being 0 difference between trial and control stores

# In[48]:


#### Over to you! Calculate the t-values for the trial months.
per_diff['t_values'] = (per_diff['percentagedifference']-0)/std
per_diff.loc[(per_diff['MONTH_ID'] > 201901), 't_values']


# In[49]:


### Find the 95th percentile of the t distribution with  the appropriate degrees of freedom to check whether 
### the hypothesis is statistically significant.

from scipy.stats import t
t.isf(0.05, df)
## 95th percentile of the t distribution


#  We can observe that the t-value is much larger than the 95th percentile value of the t-distribution for March and April.
# - **i.e. the increase in sales in the trial store in March and April is statistically greater than in the control store.**

# ### Let's create a more visual version of this by plotting:
# 1. The sales of the control store
# 2. The sales of the trial stores 
# 3. The 95th percentile value of sales of the control store.

# In[199]:


#### Control store 95th and 5th percentile
control = per_diff[['MONTH_ID', 'scaled_control_sales']]
control['MONTH_ID'] = pd.to_datetime(control['MONTH_ID'], format="%Y%m")                    
#control = control.set_index('MONTH_ID')
control['5% CI'] = control['scaled_control_sales'] * (1 - std*2)
control['95% CI'] = control['scaled_control_sales'] * (1 + std*2)
control


# In[200]:


trial = per_diff[['MONTH_ID', 'Total_Sales']]
trial['MONTH_ID'] = pd.to_datetime(trial['MONTH_ID'], format="%Y%m")
#trial = trial.set_index('MONTH_ID')
trial


# In[201]:


combinedsales = pd.merge(control, trial, left_index=False, right_index=False)
combinedsales = combinedsales.rename(columns= {'scaled_control_sales': 'control_sales', 'Total_Sales':'trial_sales'})
#combinedsales['MONTH_ID'] = pd.to_datetime(combinedsales['MONTH_ID'], format="%Y%m")
combinedsales.head()


# In[202]:


a = pd.pivot_table(combinedsales, values='control_sales', index='MONTH_ID')
b = pd.pivot_table(combinedsales, values='trial_sales', index='MONTH_ID')
c = pd.pivot_table(combinedsales, values='5% CI', index='MONTH_ID')
d = pd.pivot_table(combinedsales, values='95% CI', index='MONTH_ID')


# In[203]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(a, '-o', b, '-o', c, '--r', d, '--r')


# In[205]:


plt.xlabel('Period')
plt.ylabel('Total Amount of Sales')
plt.title('Total Sales per Month')
plt.legend(['Control Stores (77) - Sales', 'Trial Stores - Sales', 'CI 5% Control Store', 'CI 95% Control Store'])


# ## Trial Store 86

# In[56]:


storeComparison = 86

#### Over to you! Use the functions we created earlier to calculate correlations and magnitude for each potential control store

# Calculate correlations against store 86 using total sales and number of customers
corr_sales = calculateCorrelation(stores_full_obs_pre_trial, 'Total_Sales', storeComparison)
corr_ncustomers = calculateCorrelation(stores_full_obs_pre_trial, 'N_Customers', storeComparison)

# Calculate magnitude measure against store 86 using total sales and number of customers
magnitude_sales = calculateMagnitudeDistance(stores_full_obs_pre_trial, 'Total_Sales', storeComparison)
magnitude_ncustomers = calculateMagnitudeDistance(stores_full_obs_pre_trial, 'N_Customers', storeComparison)


# In[57]:


print(corr_sales.head())
print(corr_ncustomers.head())
print(magnitude_sales.head())
print(magnitude_ncustomers.head())


# In[58]:


#### Now, create a combined score composed of correlation and magnitude
corr_sales = corr_sales.rename(columns={'Correlation':'Correlation_Sales'})
magnitude_sales = magnitude_sales.rename(columns={'MagnitudeMeasure': 'Magnitude_Sales'})
corr_ncustomers = corr_ncustomers.rename(columns={'Correlation':'Correlation_NCustomers'})
magnitude_ncustomers = magnitude_ncustomers.rename(columns={'MagnitudeMeasure':'Magnitude_Ncustomers'})


# In[59]:


sales = pd.merge(corr_sales, magnitude_sales, left_index=False, right_index=False)
ncustomers = pd.merge(corr_ncustomers, magnitude_ncustomers, left_index=False, right_index=False)


# In[60]:


#### Hint: A simple average on the scores would be 0.5 * corr_measure + 0.5 * mag_measure
#### Finally, combine scores across the drivers using a simple average.
corr_measure = 0.5
# 1. Sales Table
sales['combined_score_sales'] = ((corr_measure * sales['Correlation_Sales']) + ((1-corr_measure)*sales['Magnitude_Sales']))
sales.head()


# In[61]:


# 2. N_Customers Table
ncustomers['combined_score_ncustomers'] = ((corr_measure * ncustomers['Correlation_NCustomers']) + ((1-corr_measure)*ncustomers['Magnitude_Ncustomers']))
ncustomers.head()


# In[62]:


#Now we have a score for each of total number of sales and number of customers. Let's combine the two via a simple average.
### Over to you! Combine scores across the drivers by first merging our sales scores and customer scores into a single table.
final_score = pd.merge(sales, ncustomers, left_index=False, right_index=False)
final_score = final_score.drop(['Correlation_Sales','Magnitude_Sales', 'Correlation_NCustomers', 'Magnitude_Ncustomers'], axis=1)
final_score['final_score'] = ((corr_measure * final_score['combined_score_sales']) + ((1-corr_measure)*final_score['combined_score_ncustomers']))
final_score.head()


# In[63]:


#### Select control stores based on the highest matching store
#### (closest to 1 but not the store itself, i.e. the second ranked highest store)
final_score.sort_values('final_score', ascending=False).head(10)

Looks like store 155 will be a control store for trial store 86. Again, let's check visually if the drivers are indeed similar in the period before the trial.
# In[117]:


#### Visual checks on trends based on the drivers
#We'll look at total sales first.
trial = [86]
trial_store = measures[measures.STORE_NBR.isin(trial)]
c = pd.pivot_table(trial_store, values='Total_Sales', index='MONTH_ID')

control=[155]
control_store = measures[measures.STORE_NBR.isin(control)]
d = pd.pivot_table(control_store, values='Total_Sales', index='MONTH_ID')

other = ['Other Stores']
other_store = measures[measures.store_type.isin(other)]
e = pd.pivot_table(other_store, values='Total_Sales', index='MONTH_ID')


# In[118]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(c, '-o', d, '-o')


# In[119]:


plt.plot(e, '--r')


# In[120]:


plt.xlabel('Period')
plt.ylabel('Dollar Amounts')
plt.title('Monthly Total Sales Dollar Amounts')
plt.legend(['Trial Store (77)', 'Control Store (233)', 'Other Stores'])

Great, sales are trending in a similar way.
# In[121]:


# Next, number of customers.
trial = [86]
trial_store = measures[measures.STORE_NBR.isin(trial)]
c = pd.pivot_table(trial_store, values='N_Customers', index='MONTH_ID')

control=[155]
control_store = measures[measures.STORE_NBR.isin(control)]
d = pd.pivot_table(control_store, values='N_Customers', index='MONTH_ID')


# In[122]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(c, '-o', d, '-o')


# In[124]:


plt.xlabel('Period')
plt.ylabel('Number of Customers')
plt.title('Monthly Number Of Customers')
plt.legend(['Trial Store (86)', 'Control Store (155)', 'Other Stores'])

Good, the trend in number of customers is also similar.
# In[72]:


#Let's now assess the impact of the trial on sales.


# In[125]:


#### Scale pre-trial control sales to match pre-trial trial store sales 
pre_trial_trial = stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 86]['Total_Sales'].sum()
pre_trial_control = stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 155]['Total_Sales'].sum()


# In[126]:


#### Apply the scaling factor
scalingfactor = pre_trial_trial/pre_trial_control
stores_full_obs_155 = stores_full_obs[stores_full_obs['STORE_NBR'] == 155]
stores_full_obs_155['scaled_control_sales'] = stores_full_obs_155['Total_Sales']*scalingfactor
stores_full_obs_155.reset_index(drop=True).head()


# In[127]:


#### Over to you! Calculate the percentage difference between scaled control sales and trial sales
#### Hint: When calculating percentage difference, remember to use absolute difference 
scaled_control =stores_full_obs_155[['MONTH_ID', 'scaled_control_sales']].reset_index(drop=True)

trialsales = metrics[metrics['STORE_NBR'] == 86].reset_index(drop=True)
trialsales = trialsales[['MONTH_ID', 'Total_Sales']]

per_diff = pd.merge(scaled_control, trialsales, left_index=False, right_index=False)
per_diff['percentagedifference'] = abs(per_diff.scaled_control_sales - per_diff.Total_Sales) / per_diff.scaled_control_sales
per_diff


# In[128]:


#### As our null hypothesis is that the trial period is the same as the pre-trial period, let's take the standard deviation 
#### based on the scaled percentage difference in the pre-trial period.
#### Over to you! Calculate the standard deviation of percentage differences during the pre-trial period

std = (per_diff[per_diff['MONTH_ID'] < 201902]['percentagedifference'].std())
std


# In[129]:


## degreesOfFreedom <- 7
df = 7


# In[130]:


#### Trial and control store total sales
#### Over to you! Create a table with sales by store type and month.
#### Hint: We only need data for the trial and control store.
#DONE: scaled_control_sales --> Control Sales & Total_Sales --> Trial Sales


# In[131]:


#### Over to you! Calculate the 5th and 95th percentile for control store sales.
#### Hint: The 5th and 95th percentiles can be approximated by using two standard deviations away from the mean.
#### Hint2: Recall that the variable stdDev earlier calculates standard deviation in percentages, and not dollar sales.

control = per_diff[['MONTH_ID', 'scaled_control_sales']]
control['MONTH_ID'] = pd.to_datetime(control['MONTH_ID'], format="%Y%m")                    
#control = control.set_index('MONTH_ID')
control['5% CI'] = control['scaled_control_sales'] * (1 - std*2)
control['95% CI'] = control['scaled_control_sales'] * (1 + std*2)
control


# In[132]:


trial = per_diff[['MONTH_ID', 'Total_Sales']]
trial['MONTH_ID'] = pd.to_datetime(trial['MONTH_ID'], format="%Y%m")
#trial = trial.set_index('MONTH_ID')
trial


# In[133]:


#### Then, create a combined table with columns
combinedsales = pd.merge(control, trial, left_index=False, right_index=False)
combinedsales = combinedsales.rename(columns= {'scaled_control_sales': 'control_sales', 'Total_Sales':'trial_sales'})
#combinedsales['MONTH_ID'] = pd.to_datetime(combinedsales['MONTH_ID'], format="%Y%m")
combinedsales.head()


# In[134]:


a = pd.pivot_table(combinedsales, values='control_sales', index='MONTH_ID')
b = pd.pivot_table(combinedsales, values='trial_sales', index='MONTH_ID')
c = pd.pivot_table(combinedsales, values='5% CI', index='MONTH_ID')
d = pd.pivot_table(combinedsales, values='95% CI', index='MONTH_ID')


# In[135]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(a, '-o', b, '-o', c, '--r', d, '--r')


# In[136]:


plt.xlabel('Period')
plt.ylabel('Total Amount of Sales')
plt.title('Total Sales per Month')
plt.legend(['Control Stores - Sales', 'Trial Stores - Sales', 'CI 5% Control Store', 'CI 95% Control Store'])


# ### Let's have a look at assessing this for the number of customers as well.

# In[137]:


#### This would be a repeat of the steps before for total sales
#### Scale pre-trial control customers to match pre-trial trial store customers 
pre_trial_trial = stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 86]['N_Customers'].sum()
pre_trial_control = stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 155]['N_Customers'].sum()


# In[138]:


scalingfactor = pre_trial_trial / pre_trial_control
scalingfactor


# In[139]:


stores_full_obs


# In[140]:


#### Apply the scaling factor
stores_full_obs_155['scaled_customers_control'] = stores_full_obs_155['N_Customers']*scalingfactor
stores_full_obs_155.head()


# In[141]:


stores_full_obs_155 = stores_full_obs_155[['MONTH_ID', 'scaled_customers_control']].reset_index(drop=True)
stores_full_obs_155


# In[142]:


#### Calculate the percentage difference between scaled control sales and trial sales
trial_customers = metrics[metrics['STORE_NBR'] == 86]
trial_customers = trial_customers[['MONTH_ID', 'N_Customers']].reset_index(drop=True)
trial_customers


# In[143]:


customers = pd.merge(stores_full_obs_155, trial_customers, left_index=False, right_index=False)
customers


# In[144]:


#### Calculate the percentage difference between scaled control # customers and trial # customers
customers['percentage_difference'] = abs(customers.scaled_customers_control - customers.N_Customers) / customers.scaled_customers_control
customers.head()


# In[145]:


#### As our null hypothesis is that the trial period is the same as the pre-trial period, let's take the standard deviation
#### based on the scaled percentage difference in the pre-trial period
std = (customers[customers['MONTH_ID'] < 201902])['percentage_difference'].std()
std
## degreesOfFreedom <- 7


# In[146]:


#### Trial and control store number of customers
#### Control store 95th & 5th percentile
#control = customers[['MONTH_ID', 'scaled_customers_control']]
customers['MONTH_ID'] = pd.to_datetime(customers['MONTH_ID'], format="%Y%m")                    
#control = control.set_index('MONTH_ID')
customers['5% CI'] = customers['scaled_customers_control'] * (1 - std*2)
customers['95% CI'] = customers['scaled_customers_control'] * (1 + std*2)
customers.head()


# In[147]:


#### Plotting these in one nice graph
a = pd.pivot_table(customers, values='scaled_customers_control', index='MONTH_ID')
b = pd.pivot_table(customers, values='N_Customers', index='MONTH_ID')
c = pd.pivot_table(customers, values='5% CI', index='MONTH_ID')
d = pd.pivot_table(customers, values='95% CI', index='MONTH_ID')


# In[148]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(a, '-o', b, '-o', c, '--r', d, '--r')


# In[149]:


plt.xlabel('Period')
plt.ylabel('Number of Customers')
plt.title('Total Number of Customers per Month')
plt.legend(['Control Stores - # Customers', 'Trial Stores # Customers', 'CI 5% Control Store', 'CI 95% Control Store'])

It looks like the number of customers is significantly higher in all of the three months. This seems to suggest that the trial
had a significant impact on increasing the number of customers in trial store 86 but as we saw, sales were not significantly 
higher. 

We should check with the Category Manager if there were special deals in the trial store that were may have resulted in lower 
prices, impacting the results.
# ## Trial Store 88

# In[172]:


storeComparison = 88
# Calculate correlations against store 88 using total sales and number of customers
corr_sales = calculateCorrelation(stores_full_obs_pre_trial, 'Total_Sales', storeComparison)
corr_ncustomers = calculateCorrelation(stores_full_obs_pre_trial, 'N_Customers', storeComparison)

# Calculate magnitude measure against store 88 using total sales and number of customers
magnitude_sales = calculateMagnitudeDistance(stores_full_obs_pre_trial, 'Total_Sales', storeComparison)
magnitude_ncustomers = calculateMagnitudeDistance(stores_full_obs_pre_trial, 'N_Customers', storeComparison)


# In[173]:


#### Now, create a combined score composed of correlation and magnitude
corr_sales = corr_sales.rename(columns={'Correlation':'Correlation_Sales'})
magnitude_sales = magnitude_sales.rename(columns={'MagnitudeMeasure': 'Magnitude_Sales'})
corr_ncustomers = corr_ncustomers.rename(columns={'Correlation':'Correlation_NCustomers'})
magnitude_ncustomers = magnitude_ncustomers.rename(columns={'MagnitudeMeasure':'Magnitude_Ncustomers'})


# In[174]:


sales = pd.merge(corr_sales, magnitude_sales, left_index=False, right_index=False)
ncustomers = pd.merge(corr_ncustomers, magnitude_ncustomers, left_index=False, right_index=False)


# In[175]:


#### Create a combined score composed of correlation and magnitude by merging the correlations table and the magnitudes table, 
#### for each driver.

corr_measure = 0.5
# 1. Sales Table
sales['combined_score_sales'] = ((corr_measure * sales['Correlation_Sales']) + ((1-corr_measure)*sales['Magnitude_Sales']))
sales.head()


# In[176]:


# 2. N_Customers Table
ncustomers['combined_score_ncustomers'] = ((corr_measure*ncustomers['Correlation_NCustomers']) + (1-corr_measure)*ncustomers['Magnitude_Ncustomers'])
ncustomers.head()


# In[167]:


#### Combine scores across the drivers by merging sales scores and customer scores, and compute a final combined score.
final_score = pd.merge(sales, ncustomers, right_index=False, left_index=False)
final_score = final_score[['Store1', 'Store2', 'combined_score_sales', 'combined_score_ncustomers']]
final_score.head()


# In[178]:


final_score['final_score'] = ((corr_measure * final_score['combined_score_sales']) + ((1-corr_measure)*final_score['combined_score_ncustomers']))
final_score = final_score.sort_values('final_score', ascending=False)
final_score.head()


# In[186]:


trial = [88]
trial_store = measures[measures['STORE_NBR'].isin(trial)]
c = pd.pivot_table(trial_store, values='Total_Sales', index='MONTH_ID')

control = [178]
control_store = measures[measures['STORE_NBR'].isin(control)]
d = pd.pivot_table(control_store, values='Total_Sales', index='MONTH_ID')

store_14 = [14]
control_store_14 = measures[measures['STORE_NBR'].isin(store_14)]
d1 = pd.pivot_table(control_store_14, values='Total_Sales', index='MONTH_ID')

store_134 = [134]
control_store_134 = measures[measures['STORE_NBR'].isin(store_134)]
d2 = pd.pivot_table(control_store_134, values='Total_Sales', index='MONTH_ID')

store_237 = [237]
control_store_237 = measures[measures['STORE_NBR'].isin(store_237)]
d3 = pd.pivot_table(control_store_237, values='Total_Sales', index='MONTH_ID')


# In[187]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(c, '-o', d, '-o', d1, '-o', d2, '-o', d3, '-o')


# In[189]:


plt.xlabel('Period')
plt.ylabel('Dollar Amounts')
plt.title('Monthly Total Sales Dollar Amounts')
plt.legend(['Trial Store (88)', 'Store (178)', 'Store (14)', 'Store (134)', 'Store (237)'])


# In[208]:


trial = [88]
trial_store = measures[measures['STORE_NBR'].isin(trial)]
c = pd.pivot_table(trial_store, values='N_Customers', index='MONTH_ID')

control = [178]
control_store = measures[measures['STORE_NBR'].isin(control)]
d = pd.pivot_table(control_store, values='N_Customers', index='MONTH_ID')

store_14 = [14]
control_store_14 = measures[measures['STORE_NBR'].isin(store_14)]
d1 = pd.pivot_table(control_store_14, values='N_Customers', index='MONTH_ID')

store_134 = [134]
control_store_134 = measures[measures['STORE_NBR'].isin(store_134)]
d2 = pd.pivot_table(control_store_134, values='N_Customers', index='MONTH_ID')

store_237 = [237]
control_store_237 = measures[measures['STORE_NBR'].isin(store_237)]
d3 = pd.pivot_table(control_store_237, values='N_Customers', index='MONTH_ID')


# In[209]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(c, '-o', d, '-o', d1, '-o', d2, '-o', d3, '-o')


# In[210]:


plt.xlabel('Period')
plt.ylabel('Number of Customers')
plt.title('Monthly Number of Customers')
plt.legend(['Trial Store (88)', 'Store (178)', 'Store (14)', 'Store (134)', 'Store (237)'])

We've now found store 237 to be a suitable control store for trial store 88.
After doing some visualisations, found that stores 178, 14 and 134 do not match trial store so set store 237 as control store
# In[ ]:


### Let's now assess the impact of the trial on sales.


# In[256]:


#### Scale pre-trial control store sales to match pre-trial trial store sales
pre_trial_trial = stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 88]['Total_Sales'].sum()
pre_trial_control =stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 237]['Total_Sales'].sum()


# In[257]:


scalingfactor = pre_trial_trial / pre_trial_control
scalingfactor


# In[259]:


#### Apply the scaling factor
store_full_obs_237 = stores_full_obs[stores_full_obs['STORE_NBR'] == 237]
store_full_obs_237['scaled_sales'] = store_full_obs_237['Total_Sales']*scalingfactor
store_full_obs_237 = store_full_obs_237.reset_index(drop=True)
store_full_obs_237 = store_full_obs_237[['MONTH_ID', 'N_Customers', 'scaled_sales']]
store_full_obs_237.head()


# In[261]:


trial_sales = stores_full_obs[stores_full_obs['STORE_NBR'] == 88]
trial_sales = trial_sales[['MONTH_ID', 'Total_Sales']].reset_index(drop=True)
trial_sales.head()


# In[262]:


#### Calculate the absolute percentage difference between scaled control sales and trial sales
combinedsales = pd.merge(store_full_obs_237, trial_sales, right_index=False, left_index=False)
combinedsales['perdiff'] = abs(combinedsales.scaled_sales - combinedsales.Total_Sales) / combinedsales.scaled_sales
combinedsales


# In[263]:


#### As our null hypothesis is that the trial period is the same as the pre-trial period, let's take the standard deviation 
##### based on the scaled percentage difference in the pre-trial period 
std = (combinedsales[combinedsales['MONTH_ID'] < 201902])['perdiff'].std()
std
### degreesOfFreedom <- 7


# In[264]:


#### Trial and control store total sales
#### Control store 5th and 95th percentile
combinedsales['MONTH_ID'] = pd.to_datetime(combinedsales['MONTH_ID'], format="%Y%m")
combinedsales['5% CI'] = combinedsales['scaled_sales'] * (1 - std*2)
combinedsales['95% CI'] = combinedsales['scaled_sales'] * (1 + std*2)
combinedsales


# In[248]:


#### Plotting these in one nice graph
a = pd.pivot_table(combinedsales, values='scaled_sales', index='MONTH_ID')
b = pd.pivot_table(combinedsales, values='Total_Sales', index='MONTH_ID')
c = pd.pivot_table(combinedsales, values='5% CI', index='MONTH_ID')
d = pd.pivot_table(combinedsales, values='95% CI', index='MONTH_ID')


# In[250]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(a, '-o', b, '-o', c, '--r', d, '--r')


# In[252]:


plt.xlabel('Period')
plt.ylabel('Dollar Amounts')
plt.title('Monthly Total Sales Dollar Amounts')
plt.legend(['Control Stores - Sales', 'Trial Stores - Sales', 'CI 5% Control Store', 'CI 95% Control Store'])

The results show that the trial in store 88 is significantly different to its control store in the trial period as the trial 
store performance lies outside of the 5% to 95% confidence interval of the control store in two of the three trial months.
# ###  Let's have a look at assessing this for number of customers as well.

# In[265]:


#### This would be a repeat of the steps before for total sales
#### Scale pre-trial control customers to match pre-trial trial store customers 
pre_trial_trial = stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 88]['N_Customers'].sum()
pre_trial_control = stores_full_obs_pre_trial[stores_full_obs_pre_trial['STORE_NBR'] == 237]['N_Customers'].sum()


# In[266]:


scalingfactor = pre_trial_trial / pre_trial_control
scalingfactor


# In[269]:


#### Apply the scaling factor
store_full_obs_237['scaled_customers_control'] = store_full_obs_237['N_Customers']*scalingfactor
store_full_obs_237.head()


# In[271]:


store_full_obs_237 = store_full_obs_237[['MONTH_ID', 'scaled_customers_control']].reset_index(drop=True)
store_full_obs_237


# In[272]:


#### Calculate the percentage difference between scaled control sales and trial sales
trial_customers = metrics[metrics['STORE_NBR'] == 88]
trial_customers = trial_customers[['MONTH_ID', 'N_Customers']].reset_index(drop=True)
trial_customers


# In[278]:


customers = pd.merge(store_full_obs_237, trial_customers, left_index=False, right_index=False)
customers.head()


# In[280]:


#### Calculate the percentage difference between scaled control # customers and trial # customers
customers['percentage_difference'] = abs(customers.scaled_customers_control - customers.N_Customers) / customers.scaled_customers_control
customers


# In[281]:


#### As our null hypothesis is that the trial period is the same as the pre-trial period, let's take the standard deviation
#### based on the scaled percentage difference in the pre-trial period
std = (customers[customers['MONTH_ID'] < 201902])['percentage_difference'].std()
std
## degreesOfFreedom <- 7


# In[282]:


#### Trial and control store number of customers
#### Control store 95th & 5th percentile
#control = customers[['MONTH_ID', 'scaled_customers_control']]
customers['MONTH_ID'] = pd.to_datetime(customers['MONTH_ID'], format="%Y%m")                    
#control = control.set_index('MONTH_ID')
customers['5% CI'] = customers['scaled_customers_control'] * (1 - std*2)
customers['95% CI'] = customers['scaled_customers_control'] * (1 + std*2)
customers.head()


# In[283]:


#### Plotting these in one nice graph
a = pd.pivot_table(customers, values='scaled_customers_control', index='MONTH_ID')
b = pd.pivot_table(customers, values='N_Customers', index='MONTH_ID')
c = pd.pivot_table(customers, values='5% CI', index='MONTH_ID')
d = pd.pivot_table(customers, values='95% CI', index='MONTH_ID')


# In[284]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(a, '-o', b, '-o', c, '--r', d, '--r')


# In[285]:


plt.xlabel('Period')
plt.ylabel('Number of Customers')
plt.title('Total Number of Customers per Month')
plt.legend(['Control Stores - # Customers', 'Trial Stores # Customers', 'CI 5% Control Store', 'CI 95% Control Store'])

Total number of customers in the trial period for the trial store is significantly higher than the control store for two out 
of three months, which indicates a positive trial effect.
# # Conclusion
# Good work! We've found control stores 233, 155, 237 for trial stores 77, 86 and 88 respectively.
# 
# The results for trial stores 77 and 88 during the trial period show a significant difference in at least two of the three trial months but this is not the case for trial store 86. We can check with the client if the implementation of the trial was different in trial store 86 but overall, the trial shows a significant increase in sales. Now that we have finished our analysis, we can prepare our presentation to the Category Manager.

# In[ ]:




