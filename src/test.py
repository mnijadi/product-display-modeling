""" testing the local API of the model"""

import requests
import pandas as pd

df = pd.read_csv('../data/raw/new_Base_CDM_balanced_V2.csv', delimiter=';', header=1)

# data example
product = df.drop('Display', axis=1).iloc[0].to_dict()
display = df['Display'].iloc[0]
print(product)

url = 'http://localhost:9696/predict'
response = requests.post(url, json=product)
print('actual:', display)
print('model response:', response.json())