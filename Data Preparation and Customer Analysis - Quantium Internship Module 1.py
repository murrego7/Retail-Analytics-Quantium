#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

transactional = pd.read_csv('QVI_transaction_data.csv', sep=';')


# # Exploratory data analysis

# In[2]:


transactional.info()
# Are columns we would expect to be numeric in numeric form and date columns are in date format?
#Date is a int, shouldn't be


# In[3]:


transactional.head(10)


# In[4]:


from datetime import date, timedelta
start = date(1899,12,30)

new_date_format = []

for date in transactional["DATE"]:
    delta = timedelta(date)
    new_date_format.append(start + delta)


# In[5]:


transactional["DATE"] = pd.to_datetime(pd.Series(new_date_format))
print(transactional["DATE"].dtype)


# In[6]:


transactional.head()


# In[7]:


#Generate a summary of the PROD_NAME column.
transactional['PROD_NAME'].describe()


# In[8]:


#Looks like we are definitely looking at potato chips but how can we check that these are all chips? We can do some basic text
#analysis by summarising the individual words in the product name.
transactional.PROD_NAME.str.split(expand=False)


# In[9]:


#There are salsa products in the dataset but we are only interested in the chips category, so let's remove these.

transactional = transactional[~transactional["PROD_NAME"].str.contains(r"[Ss]alsa")]
transactional.reset_index(inplace=True)


# In[10]:


#transactional.PROD_NAME.str.split(expand=True)


# In[11]:


# Create a variable called "PACK_SIZE"
transactional['PACK_SIZE'] = transactional.PROD_NAME.str.extract("(\d+)")
transactional['PACK_SIZE'] = pd.to_numeric(transactional.PACK_SIZE)
transactional['PACK_SIZE']


# In[12]:


# Create a variable called "BRAND_NAME" we can use the first word in PROD_NAME to work out the brand name
transactional['BRAND_NAME'] = transactional['PROD_NAME'].str.split(' ').str[0]
transactional['BRAND_NAME']


# In[13]:


#As we are only interested in words that will tell us if the product is chips or not, let's remove all words with digits 
#and special characters such as '&' from our set of product words

transactional.PROD_NAME = transactional.PROD_NAME.str.strip()     
transactional.PROD_NAME = transactional.PROD_NAME.str.replace(' ', '_')
transactional.PROD_NAME = transactional.PROD_NAME.str.replace(r"[^a-zA-Z\d\_]+", "")    
transactional.PROD_NAME = transactional.PROD_NAME.str.replace(r"[^a-zA-Z\d\_]+", "")
transactional.PROD_NAME = transactional.PROD_NAME.str.replace("\d+", "")
transactional.PROD_NAME = transactional.PROD_NAME.str.replace("g", "")


# In[14]:


#Column after text transformation
transactional.PROD_NAME.str.split(expand=False)


# In[15]:


#Let's look at the most common words by counting the number of times a word appears and sorting them by this frequency 
#in order of highest to lowest frequency
from collections import Counter
most_common = Counter(" ".join(transactional.PROD_NAME).split()).most_common(5)
most_common


# In[16]:


#After dropping rows containg "salsa" word, we have decreased the # of rows from 264836 to 246742
transactional.info()


# In[17]:


# Summarise the data to check for nulls and possible outliers
print(transactional.STORE_NBR.describe())
print(transactional.LYLTY_CARD_NBR.describe())
print(transactional.TXN_ID.describe())
print(transactional.PROD_NBR.describe())
print(transactional.PROD_QTY.describe()) #We catch an outlier, as the max value 200 refers to a transaction where 200 units
                                        #were bought.
                                        
print(transactional.TOT_SALES.describe())


# In[18]:


#### Let's see if the customer has had other transactions
# Use a filter to see what other transactions that customer made.
transactional[(transactional['LYLTY_CARD_NBR'] == 226000)]
#RESULT: It looks like this customer has only had the two transactions over the year and is not an ordinary retail customer. 
#The customer might be buying chips for commercial purposes instead. 


# In[19]:


##We'll remove this loyalty card number from further analysis
transactional = transactional[transactional.LYLTY_CARD_NBR != 226000]
transactional.reset_index(inplace=True)


# In[20]:


## Max transaction now is 5, seems normal
transactional.PROD_QTY.describe()


# In[21]:


#Now, let's look at the number of transaction lines over time to see if there are any obvious data issues such as missing data.
#### Count the number of transactions by date
# Create a summary of transaction count by date.
transactional.DATE.nunique()


# In[22]:


#There's only 364 rows, meaning only 364 dates which indicates a missing date. 
pd.date_range(start= '2018-07-01', end= '2019-06-30').difference(transactional.DATE)
##RESULT: CHRISTMAS IS MISSING, MOST OF THE STORES ARE CLOSED


# *Let's create a sequence of dates from 1 Jul 2018 to 30 Jun 2019 and use this to create a chart of number of transactions over time to find the missing date*

# In[23]:


#### Create a sequence of dates and join this the count of transactions by date
a = pd.pivot_table(transactional, values= 'TOT_SALES', index='DATE', aggfunc='sum')
a


# In[24]:


### Let's create a sequence of dates from 1 Jul 2018 to 30 Jun 2019
b = pd.DataFrame(index=pd.date_range(start= '2018-07-01', end= '2019-06-30'))
b['TOT_SALES']=0
b


# In[25]:


#Join the sequence of dates from 1 Jul 2018 to 30 Jun 2019 onto the data to fill in the missing day.
c = a + b
c.fillna(0, inplace = True)
c.index.name = 'DATE'
c


# In[26]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
plt.figure()
plt.plot(c,'-',)


# In[27]:


x = plt.gca() #Get the current axis
x.set_title('Number of Transactions Over Time')
x.set_ylabel('Number of Transactions')
#We can see that there is an increase in purchases in December and a break in late December.


# In[28]:


#### Filter to December and look at individual days
december = c.loc['2018-12-01':'2018-12-31']
december.head()


# In[29]:


plt.figure()
plt.plot(december,'-',)
#We can see that the increase in sales occurs in the lead-up to Christmas and that there are zero sales on Christmas day itself.
#This is due to shops being closed on Christmas day.


# In[30]:


x = plt.gca().xaxis

# rotate the tick labels for the x axis
for item in x.get_ticklabels():
    item.set_rotation(45)
    
# adjust the subplot so the text doesn't run off the image
plt.subplots_adjust(bottom=0.25)


# In[31]:


x = plt.gca()
x.set_ylabel('Number of transactions')
x.set_title('Transactions Over December')


# In[32]:


#Now that we are satisfied that the data no longer has outliers, we can move on to creating other features such as brand 
#of chips or pack size from PROD_NAME


# In[33]:


#### Pack size was created, let's analize it
transactional.PACK_SIZE.describe()
#The largest size is 380g and the smallest size is 70g


# In[34]:


##Plot a histogram showing the number of transactions by pack size.


# In[35]:


newdf = pd.pivot_table(transactional, values= 'TOT_SALES', index='PACK_SIZE', aggfunc='sum')
newdf.head()


# In[36]:


df1 = newdf.reset_index()


# In[37]:


a = df1.plot(x= "PACK_SIZE", y= "TOT_SALES", kind="bar", width=2, color='royalblue', alpha=1, title = 'Transactions By Pack Size')


# In[38]:


#### Brand Name was created, let's analize it and check if the results are reasonable
transactional.BRAND_NAME.unique()


# In[39]:


#Some of the brand names look like they are of the same brands - such as RED and RRD, which are both Red Rock Deli chips
#Let's combine these together
transactional.BRAND_NAME.replace('Red', 'RRD', inplace=True)
transactional.BRAND_NAME.replace('Dorito', 'Doritos', inplace=True)
transactional.BRAND_NAME.replace('Smith', 'Smiths', inplace=True)
transactional.BRAND_NAME.replace('Snbts', 'Sunbites', inplace=True)
transactional.BRAND_NAME.replace('NCC', 'Natural', inplace=True)
transactional.BRAND_NAME.replace('Grain', 'GrnWves', inplace=True)
transactional.BRAND_NAME.replace('WW', 'Woolworths', inplace=True)
transactional.BRAND_NAME.replace('Infzns', 'Infuzions', inplace=True)


transactional.BRAND_NAME.unique()


# # Examining customer data

# In[40]:


customer = pd.read_csv('QVI_purchase_behaviour.csv')
#Do some basic summaries of the dataset, including distributions of any key columns.
customer.head()


# In[41]:


customer.info()


# In[42]:


customer.PREMIUM_CUSTOMER.describe()


# In[43]:


plt.figure()
customer.PREMIUM_CUSTOMER.value_counts().plot(kind='bar', width=0.9, title='Distribution of Types of Premium Customer')


# In[44]:


#we can be sure that no duplicates were created. This is because we created `data` by setting a left join which means 
#take all the rows in `transactionData` and find rows with matching values in shared columns and then joining the details 
#in the first mentioned table
data = transactional.merge(customer, how='left')


# In[45]:


# Let's also check if some customers were not matched on by checking for nulls (LYLTY_CARD_NBR)
# See if any transactions did not have a matched customer.
data.info()


# # Data analysis on customer segments 
# *We can define some **metrics of interest** to the client:*
# 1. Who spends the most on chips (total sales), describing customers by lifestage and how premium their general purchasing behaviour is
# 2. How many customers are in each segment
# 3. How many chips are bought per customer by segment
# 4. What's the average chip price by customer segment
# 
# *We could also ask our data team for more information:*
# - The customer's *total spend over the period* and *total spend for each transaction* to **understand what proportion of their grocery spend is on chips.**
# 
# - *Proportion of customers in each customer segment overall* to **compare against** the *mix of customers who purchase chips*

# # Who spends the most on chips **(total sales)**, describing customers by:
I. Lifestage 
II. How premium their general purchasing behaviour
# In[46]:


#Let's start with calculating total sales by LIFESTAGE and PREMIUM_CUSTOMER, that is calculate the summary of sales
#by those dimensions
grouped_sales = pd.DataFrame(data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].agg(['sum']))
grouped_sales.sort_values(ascending=False, by='sum')
grouped_sales.rename(columns = {'sum':'TOT_SALES'})
## Total sales by LIFESTAGE and PREMIUM_CUSTOMER


# In[47]:


plt.figure()
plot0 = grouped_sales["sum"].sort_values().plot.barh(figsize=(16,7), title = 'Total Sales by Customer Segment')


# In[48]:


import matplotlib.pyplot as plt
plot = pd.DataFrame(data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum())
plot.unstack().plot(kind = 'barh', stacked = True, figsize = (16,6), title = 'Total Sales by Customer Segment')
#Sales are coming mainly from: 
# 1. Budget - older families
# 2. Mainstream - young singles/couples
# 3. Mainstream - retirees


# # Let's see if the higher sales are due to there being more customers who buy chips

# ### 2. How many customers are in each segment
# Let's see if the higher sales are due to there being more customers who buy chips
# In[49]:


#### Number of customers by LIFESTAGE and PREMIUM_CUSTOMER
#Calculate the summary of number of customers by those dimensions and create a plot:
nunique_customers = data.groupby(["LIFESTAGE","PREMIUM_CUSTOMER"]).LYLTY_CARD_NBR.nunique().sort_values(ascending=False)
nunique_customers = pd.DataFrame(nunique_customers)
nunique_customers.rename(columns = {'LYLTY_CARD_NBR' : 'Number_Customers'}, inplace= True)
nunique_customers


# In[50]:


plt.figure()
nunique_customers = data.groupby(["LIFESTAGE", "PREMIUM_CUSTOMER"])["LYLTY_CARD_NBR"].nunique().sort_values(ascending=False)
pd.DataFrame(nunique_customers)
nunique_customers.sort_values().plot.barh(figsize=(17,7), title = 'Number of Customers by Segment')


# In[51]:


import matplotlib.pyplot as plt
plot = pd.DataFrame(data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).LYLTY_CARD_NBR.sum())
plot.unstack().plot(kind = 'barh', stacked = True, figsize = (16,6), title = 'Number of Customers by Segment')
# Customers who buy the larger amounts of chips:
# 1. Mainstream - young singles/couples
# 2. Mainstream - retirees
# (This contributes to there being more sales to these customer segments)

# BUT this is not a major driver for the BUDGET - OLDER FAMILIES SEGMENT


# # How many chips are bought per customer by segment

# In[52]:


# Higher sales may also be driven by more units of chips being bought per customer

#total quantity of products that were bought
PROD_QUANTIY = data.groupby(["PREMIUM_CUSTOMER", "LIFESTAGE"])['PROD_QTY'].sum()

#### Number of customers by LIFESTAGE and PREMIUM_CUSTOMER
unique_customers = data.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE'])['LYLTY_CARD_NBR'].nunique()

#### Average number of units per customer by LIFESTAGE and PREMIUM_CUSTOMER
avg_units = PROD_QUANTIY/unique_customers
avg_units = pd.DataFrame(avg_units, columns= {'Avg_unit_customer'})
avg_units.sort_values(by='Avg_unit_customer', ascending = False)


# In[53]:


avgUnitsPlot = pd.DataFrame(data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum() / data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).LYLTY_CARD_NBR.nunique())
avgUnitsPlot.unstack().plot(kind = 'bar', figsize = (12, 7), title = 'Average Unit by Customer Segment')
plt.ylabel('Average Number of Units')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)

x = plt.gca().xaxis

# rotate the tick labels for the x axis
for item in x.get_ticklabels():
    item.set_rotation(45)
    
# adjust the subplot so the text doesn't run off the image
plt.subplots_adjust(bottom=0.25)

#Older families and young families in general buy more chips per customer


# # What's the average chip price by customer segment

# In[54]:


# Let's also investigate the average price per unit chips bought for each customer segment as this is also a driver 
#of total sales.

#total quantity of products that were bought
PROD_QUANTIY = data.groupby(["PREMIUM_CUSTOMER", "LIFESTAGE"])['PROD_QTY'].sum()

#### Total Sales by LIFESTAGE and PREMIUM_CUSTOMER
Total_sales = data.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE'])['TOT_SALES'].sum()

#### Average price per unit 
avg_units_chips = PROD_QUANTIY/Total_sales
avg_units_chips


# In[55]:


avgUnitsPricePlot = pd.DataFrame(data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).TOT_SALES.sum() / data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).PROD_QTY.sum())
avgUnitsPricePlot.unstack().plot(kind = 'bar', figsize = (12, 7), title = 'Average Unit by Customer Segment')
plt.ylabel('Average Sale Price')                              
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 2)

x = plt.gca().xaxis

# rotate the tick labels for the x axis
for item in x.get_ticklabels():
    item.set_rotation(45)
    
# adjust the subplot so the text doesn't run off the image
plt.subplots_adjust(bottom=0.25)

#Mainstream midage and young singles and couples are more willing to pay more per packet of chips compared to their budget 
#and premium counterparts. 

#This may be due to premium shoppers being more likely to buy healthy snacks and when they buy chips, this is mainly 
#for entertainment purposes rather than their own consumption.

#This is also supported by there being fewer premium midage and young singles and couples buying chips compared to their 
#mainstream counterparts.


# # As the difference in average price per unit isn't large, we can check if this difference is statistically different.
Perform an independent t-test between mainstream vs premium and budget midage and young singles and couples
Mainstream Vs Nonmainstream (midage & young singles/couples)
# In[56]:


data['price_unit'] = data['TOT_SALES']/data['PROD_QTY']


# In[57]:


from scipy.stats import ttest_ind

# Group our data into mainstream and non-mainstream 
mainstream = data['PREMIUM_CUSTOMER'] == "Mainstream"
nonmainstream = (data['PREMIUM_CUSTOMER'] == "Budget") | (data['PREMIUM_CUSTOMER'] == "Premium")

# Filter only midage & young single/couples
midage_young = (data['LIFESTAGE'] == "MIDAGE SINGLES/COUPLES") | (data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES')

#T-test:
a = data[midage_young & mainstream]["TOT_SALES"]
b = data[midage_young & nonmainstream]["TOT_SALES"]
stat, pval = ttest_ind(a.values, b.values, equal_var=False)

print(pval)
pval < 0.0000001

The t-test results in a p-value of 1.83e-237, the unit price for mainstream, young and mid-age singles and couples ARE SIGNIFICANTLY higher than that of budget or premium, young and midage singles and couples.
# ## We might want to target customer segments that contribute the most to sales to retain them or further increase sales. Let's look at Mainstream - young singles/couples. For instance, let's find out if they tend to buy a particular brand of chips

# In[58]:


#### Deep dive into Mainstream, young singles/couples 
#### Work out of there are brands that these two customer segments prefer more than others
# You could use a technique called affinity analysis or a-priori analysis


# ## Affinity to Brand

# In[59]:


newdf0 = data[(data['PREMIUM_CUSTOMER'] == "Mainstream") & (data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES')]
targetBrand = newdf0.loc[:, ['BRAND_NAME', 'PROD_QTY']]
targetBrand['Target Brand Affinity'] = targetBrand['PROD_QTY']/targetBrand['PROD_QTY'].sum()
targetBrand = pd.DataFrame(targetBrand.groupby('BRAND_NAME')['Target Brand Affinity'].sum())
targetBrand.sort_values(by='Target Brand Affinity', ascending = False).head()


# In[60]:


newdf1 = data[(data['PREMIUM_CUSTOMER'] != 'Mainstream') & (data['LIFESTAGE'] != 'YOUNG SINGLES/COUPLES')]
nonTargetBrand = newdf1.loc[:, ['BRAND_NAME', 'PROD_QTY']]
nonTargetBrand['Nontarget Brand Affinity'] = nonTargetBrand['PROD_QTY']/nonTargetBrand['PROD_QTY'].sum()
nonTargetBrand = pd.DataFrame(nonTargetBrand.groupby('BRAND_NAME')['Nontarget Brand Affinity'].sum())
nonTargetBrand.sort_values(by='Nontarget Brand Affinity', ascending=False).head()


# In[66]:


merged_ = pd.merge(targetBrand, nonTargetBrand, left_index=True, right_index=True)
merged_['Affinity To Brand'] = merged_['Target Brand Affinity']/merged_['Nontarget Brand Affinity']
merged_.sort_values(by = 'Affinity To Brand', ascending=False)
#Mainstream young singles/couples are more likely to purchase Tyrrells chips compared to other brands.

# Insights: 
-Mainstream Young Single Couples prefer to buy Kettle, Smiths, Doritos, Pringles, Smiths and Infuzions Brand Chips. Being Kettle Chips the most prefered (19% of the total transactions).
-These segments prefer Tyrells and Twisties more than the rest of the customers (1.22 and 1.21 times respectively).
# ## Pack Size Insights

# In[62]:


targetSize = newdf0.loc[:, ['PACK_SIZE', 'PROD_QTY']]
targetSize['Target Size Affinity'] = targetSize['PROD_QTY']/targetSize['PROD_QTY'].sum()
targetSize = pd.DataFrame(targetSize.groupby('PACK_SIZE')['Target Size Affinity'].sum())
targetSize.sort_values(by='Target Size Affinity', ascending = False).head()


# In[63]:


nontargetSize = newdf1.loc[:, ['PACK_SIZE', 'PROD_QTY']]
nontargetSize['Nontarget Size Affinity'] = nontargetSize['PROD_QTY']/nontargetSize['PROD_QTY'].sum()
nontargetSize = pd.DataFrame(nontargetSize.groupby('PACK_SIZE')['Nontarget Size Affinity'].sum())
nontargetSize.sort_values(by='Nontarget Size Affinity', ascending = False).head()


# In[67]:


merged = pd.merge(targetSize, nontargetSize, left_index=True, right_index=True)
merged['Affinity To Pack Size'] = merged['Target Size Affinity']/merged['Nontarget Size Affinity']
merged.sort_values(by = 'Affinity To Pack Size', ascending=False).head()

# Insights: 
-175 gr packs are the most preferred for mainstream young single/couples.
-These segments are more likely to buy 270 gr packs respect the other customers.
# # Conclusions
# 1. Budget - older families (156863.75),  Mainstream - young singles/couples (147582.20) and Mainstream - retirees (145168.95) spends the most on chips
# 
# 2. There's a high number of customers who belong to the segment Mainstream - young singles/couples (7917 persons), the second place is composed for Mainstream - Retirees. This could explain the total impact in sales.
# 3. Mainstream - Older Families buy on average 9.25 chips units, they are also the segment who buy more on average. 
# 4. Mainstream midage and young singles/couples are more willing to pay more per packet of chips compared to their budget  and premium counterparts (more than 4 dollars).
# 

# In[ ]:





# In[ ]:




