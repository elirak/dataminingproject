import pandas as pd
from apyori import apriori
from apyori import apriori
from mlxtend.frequent_patterns import apriori, association_rules

table = pd.read_csv('data.csv')  #//Importing Dataset from system


table.head()   #// to check the header

data=table.merge(table, how = "right")

print(data.state.unique())

data['product_name']=data['product_name'].str.strip()

data.dropna(axis=0, subset=['orders_id'], inplace=True)
data['orders_id'] = data['orders_id'].astype('str')
data = data[~data['orders_id'].str.contains('C')]


basket_Pa = ((data['state'] =='NJ')
             .groupby(['orders_id', 'product_name'])['products_quantity']
             .sum().unstack().reset_index().fillna(0)
             .set_index('orders_id'))



# Defining the hot encoding function to make the data suitable
# for the concerned libraries
def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1

basket_encoded = basket_Pa.applymap(hot_encode)
basket_Pa = basket_encoded


frq_items = apriori(basket_Pa, min_support = 0.05, use_colnames = True)


rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())
#association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
#association_results = list(association_rules)

#print(len(association_rules))
#print(association_rules[0])

#for item in association_rules:

    # first index of the inner list
 #   pair = item[0]
  #  items = [x for x in pair]
   # print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    #print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

   # print("Confidence: " + str(item[2][0][2]))
   # print("Lift: " + str(item[2][0][3]))
   # print("=====================================")