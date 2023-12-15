import pandas as pd
import numpy as np

df = pd.read_csv('/home/ml/project/flow/datasets/data_processed.csv')

l = int(len(df)*0.7)
train = df[:l]
test = df[l+1:]

train.to_csv('/home/ml/project/flow/datasets/data_train.csv')
test.to_csv('/home/ml/project/flow/datasets/data_test.csv')
