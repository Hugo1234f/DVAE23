import math

# List of the K values to test against
ks = []
ks += list(range(1,30,2))
#-------------------------------------

# Calculate Euclidian distance
def l2(v1, v2):
    v1 = list(v1)
    v1 = [int(x) for x in v1]

    v2 = list(v2)
    v2 = [int(x) for x in v2]

    euclidian = 0
    for i in range(len(v1)):
        euclidian += (v1[i]-v2[i])**2
    return math.sqrt(euclidian)

# Evaluate KNN with a given K
def evaluate(k):
    corrects = 0
    for i in range(len(dataset)):
        corrects += classify(dataset, i, k)
        
    percentage = (corrects/len(dataset)) * 100
    print(f"\t{percentage:7.3f}%, \t({corrects:3}/{len(dataset)})")

#Take the majority vote
def majority_vote(distances, correct_label):
    corrects = 0
    for dist in distances:
        if dist["label"] == correct_label:
            corrects += 1

    return 1 if corrects >= (len(distances)-1)/2 else 0

#Classify a digit.
def classify(dataset, unknown_id, n=5):    
    distances = []
    for i in range(len(dataset)):
        if i == unknown_id: continue
        distances.append({"distance": l2(dataset[unknown_id]["data"], dataset[i]["data"]), "label": dataset[i]["label"]}) # calculate euclidian distance

    distances = sorted(distances, key=lambda x: x["distance"]) #sort by smallest distances
    classification = majority_vote(distances[:n], dataset[unknown_id]['label']) # evaluate the n closest neighbours
    return classification

#preprocess data
raw_data = ""
raw_lines = []

print("Reading data")
with open("./digits.txt", "r") as file:
    raw_data = file.read()
    raw_lines = raw_data[:-1].split('\n')

#assign labels 
print("Assigning labels")
dataset = []
for i in range(len(raw_lines)):
    label = 0

    if(i < 50):     label = 0
    elif(i < 100):  label = 1
    else:           label = 8

    rows = [raw_lines[i][j*32:(j+1)*32] for j in range(32)]
    rows_int = [[int(c) for c in row] for row in rows]

    row_density = sum([sum(row) for row in rows_int])/32
    col_density = sum([sum(col) for col in zip(*rows_int)])/32
    
    dataset.append({"data": [row_density, col_density], "label": label})

#Run the evaluation
print("----------")
print(f"Evaluating: k={ks}")
for k in ks:
    print(f"k={k}:", end='', flush=True)
    evaluate(k)
