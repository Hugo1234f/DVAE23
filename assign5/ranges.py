import pandas as pd

data = pd.read_csv('./points.csv', index_col=False)

ones = data.loc[data['Label'] == 1]
zeros = data.loc[data['Label'] == 0]

print('Label = 1:')
print(ones.describe())

print('\nLabel = 0:')
print(zeros.describe())