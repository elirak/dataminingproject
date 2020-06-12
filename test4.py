import pandas as pd
from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
data=pd.read_csv('dataset.csv')
data.head()
data["product_name"] = data["product_name"].astype(str)
data["orders_id"] = data["orders_id"].astype(str)
data = data.dropna()

data_ny=data[data['state']=='NY']

data_ny = (data_ny.groupby(["orders_id","product_name"])["products_quantity"].sum()
           .sum().unstack().reset_index().fillna(0)
           .set_index('orders_id'))
print(data_ny)







