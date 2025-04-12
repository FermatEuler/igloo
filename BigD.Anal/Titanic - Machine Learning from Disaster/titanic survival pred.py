import numpy as np
import pandas as pd
import os

file_path = r'C:\Users\99kos\pythonProject\BigD.Anal\Titanic - Machine Learning from Disaster'

train_df = pd.read_csv(file_path+r'\train.csv')
test_df = pd.read_csv(file_path+r'\test.csv')

print("Training data shape:",train_df.shape)
print("Test data shape:",test_df.shape)
print("====================================")

print("\nTraining data info:")
train_df.info()
print("====================================")

print("\nTraining data statistics:")
print(train_df.describe())