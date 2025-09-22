import pandas as pd
import math


ks = [x for x in range(1, 50, 2)]

def euclidian_distance(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i]-v2[i])**2
    return math.sqrt(sum)

# Read data
csv_data = pd.read_csv('./abalone.csv', index_col=0)
attributes = csv_data.index.name.split(' ')

# Organize data
processed_data = []
for abalone in csv_data.index:
    processed_data.append(dict(zip(attributes, abalone.split(' '))))

print(processed_data[0])