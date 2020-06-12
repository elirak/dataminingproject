import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df=pd.read_csv('dataset.csv')
df.head()

#data cleaning
df['product_name']=df['product_name'].str.strip() #remove space
df.dropna(axis=0, subset=['orders_id'], inplace=True) #remove duplicate order_id
df['orders_id']=df['orders_id'].astype('str') #converting nr to string
df=df[~df['orders_id'].str.contains('C')]
df.head()

print(df['state'].value_counts())
#print(df.shape)
mybasket=(df[df['state']=='NY']
        .groupby(['orders_id','product_name'])
        .sum().unstack().reset_index().fillna(0)
        .set_index('orders_id'))
mybasket.head()

def my_encode_units(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

my_basket_sets=mybasket.applymap(my_encode_units)
my_basket_sets.drop('POSTAGE', inplace=True, axis=1)


my_freuqent_items=apriori(my_basket_sets, min_support=0.04, use_colnames=True)
my_rules=association_rules(my_freuqent_items, metric='list', min_threshold=1)
my_rules.head(100)

print(my_rules)






