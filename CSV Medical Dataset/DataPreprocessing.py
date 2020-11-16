import pandas as pd
from sklearn import preprocessing

def data_preprocess(df):
   categoricals = df.select_dtypes(include='object')
   categoricals = categoricals.astype(str)
   
   label_ = preprocessing.LabelEncoder()
   categoricals = categoricals.apply(label_.fit_transform)
   
   oh = preprocessing.OneHotEncoder()
   encoded_data = oh.fit(categoricals).transform(categoricals)
   encoded_data=pd.DataFrame(encoded_data.todense())
   encoded_data.reset_index(drop=True, inplace=True)
   
   original_numeric = df.select_dtypes(include='number')
   original_numeric.reset_index(drop=True, inplace=True)
   
   data = pd.concat([original_numeric, encoded_data], axis=1)
   
   return data

