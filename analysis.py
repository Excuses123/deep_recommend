import pandas as pd


result = pd.read_csv("./output3/test_result_2.txt", sep = "\t")
result['label'] = result['label'].astype(str)
result['pred_list'] = result['pred_list'].map(lambda x: x.split(","))
result['ind'] = result.apply(lambda row: row.pred_list.index(row.label) if row.label in row.pred_list else -1, axis = 1)
result['is_in'] = result['ind'].map(lambda x: 1 if x != -1 else 0)

a = result.groupby(['ind'], as_index = False)['label'].agg({'cnt':'count'})
a['rate'] = a['cnt'] / result.shape[0]
print(a)
print("mean :",result[result.ind != -1]['ind'].mean())
print("in top20 rate: ",result.is_in.mean())
print("count: ",result.is_in.sum())

'''
data = pd.read_parquet("./data/gamebox_20191205.snappy.parquet")
result['pred_list2'] = ','.join(data['gameid'].value_counts().reset_index().head(500)['index'].astype(str).tolist())
result['pred_list2'] = result['pred_list2'].map(lambda x: x.split(","))
result['ind2'] = result.apply(lambda row: row.pred_list2.index(row.label) if row.label in row.pred_list2 else -1, axis = 1)
result['is_in2'] = result['ind2'].map(lambda x: 1 if x != -1 else 0)

# a = result.groupby(['ind2'], as_index = False)['label'].agg({'cnt':'count'})
# a['rate'] = a['cnt'] / result.shape[0]
# print(a)
print("mean :",result[result.ind2 != -1]['ind2'].mean())
print("in top20 rate: ",result.is_in2.mean())
print("count: ",result.is_in2.sum())
'''




