import pandas as pd

sample_df = pd.read_pickle("sample_df.pkl")
test_data = sample_df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
test_data_sample = test_data.sample(3).to_dict('list')
print(test_data_sample)