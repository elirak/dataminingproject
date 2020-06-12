import mlxtend
import pandas as pd

from apyori import apriori


data=pd.read_csv("data.csv")

#table.head()
#data=table['orders_id': 'product_name']


records=[]

for i in range(1000000):
    records.append([str(data.values[i,j]) for j in range(2)])

    assocation_rules=apriori(records, min_support=0.06, min_confidence=0.08, min_length=5)
    assocation_results=list(assocation_rules)

    print(len(assocation_results))
    df_results=(pd.DataFrame(assocation_results))
    df_results.head()
    support = df_results.support
    first_values = []
    second_values = []
    third_values = []
    fourth_value = []
    for i in range(df_results.shape[0]):
        single_list = df_results['ordered_statistics'][i][0]
        first_values.append(list(single_list[0]))
        second_values.append(list(single_list[1]))
        third_values.append(single_list[2])
        fourth_value.append(single_list[3])
        lhs = pd.DataFrame(first_values)
        rhs = pd.DataFrame(second_values)
        confidance = pd.DataFrame(third_values, columns=['Confidance'])
        lift = pd.DataFrame(fourth_value, columns=['lift'])
        df_final = pd.concat([lhs, rhs, support, confidance, lift], axis=1)
        #print(df_final)

        df_final.fillna(value=' ', inplace=True)
        df_final.columns = ['lhs', 1, 2, 'rhs', 'support', 'confidance', 'lift']
        df_final['lhs'] = df_final['lhs'] + str(", ") + df_final[1] + str(", ") + df_final[2]
        df_final.drop(columns=[1, 2], inplace=True)
        df_final.head()
