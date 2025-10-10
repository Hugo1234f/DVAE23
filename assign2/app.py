import pandas as pd
import math


ks = [x for x in range(1, 50, 2)]

def distance(p1, p2):
    sum = 0
    attrs = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    for attr in attrs:
        sum += (float(p1[attr]) - float(p2[attr]))**2
    return math.sqrt(sum)


# Estimate rings
def classify(dataset, test_i, k):
    test_point = dataset[test_i]
    distances = []
    for i in range(len(dataset)):
        if i != test_i:
            distances.append({'index': i, 'dist': distance(dataset[i], test_point), 'rings': dataset[i]['Rings']})

    knn = sorted(distances, key=lambda x: x['dist'])[:k]
    #print(knn)

    #Aggregate rings
    rings = 0
    for ki in knn:
        rings += int(ki['rings'])
    rings /= k

    return round(rings)

# Read data
csv_data = pd.read_csv('./abalone.csv', index_col=0)
attributes = csv_data.index.name.split(' ')

# Organize data
processed_data = []
for abalone in csv_data.index:
    processed_data.append(dict(zip(attributes, abalone.split(' '))))

print('---')
print('Testing...')




def test_k(k = 5):
    print(f"k = {k}:", end='', flush=True)
    estimate_errors = []
    for i in range(len(processed_data)):
        est = abs(classify(processed_data, i, k) - int(processed_data[i]['Rings']))
        estimate_errors.append(est)
    
    estimate = sum(estimate_errors)/len(estimate_errors)
    print(f"\t{estimate:7.3f}")
    return estimate

errs = []
for k in ks:
    errs.append(test_k(k))
print('Average error: ' + str(sum(errs)/float(len(errs))))