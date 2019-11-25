import csv
from sklearn import metrics

dataset = []
actual_values = []
predicted_values = []

with open('ConceptLearning.csv') as csvFile:
    dataset = [line for line in csv.reader(csvFile)]

split_percentage = int(0.9 * len(dataset))
train = dataset[:split_percentage]
test = dataset[split_percentage:]

print("\nOUTLOOK=> Sunny=1 Overcast=2 Rain=3\n TEMPERATURE=> Hot=1 Mild=2 Cool=3\n HUMIDITY=> High=1 Normal=2\n WIND=> Weak=1 Strong=2\n TARGET CONCEPT:PLAY TENNIS=> Yes=10 No=5\n")

p_y = 0
for row in train:
    if row[-1] == '10':
        p_y += 1
p_n = len(train) - p_y

for t in test:
    prod_y = p_y / len(train)
    prod_n = p_n / len(train)

    for i in range(len(t) - 1):
        cpy = 0
        cpn = 0
        for j in range(len(train)):
            if t[i] == train[j][i] and train[j][-1] == "10":
                cpy += 1
            if t[i] == train[j][i] and train[j][-1] == "5":
                cpn += 1

        prod_y *= cpy / p_y
        prod_n *= cpn / p_n

    if prod_y > prod_n:
        actual_values.append(t[-1])
        predicted_values.append("10")

    else:
        actual_values.append(t[-1])
        predicted_values.append("5")

print("Actual Values", actual_values)
print("Predicted Values", predicted_values)
print("Accuracy : ", metrics.accuracy_score(actual_values, predicted_values) * 100, "%")


"""
Output:

OUTLOOK=> Sunny=1 Overcast=2 Rain=3
 TEMPERATURE=> Hot=1 Mild=2 Cool=3
 HUMIDITY=> High=1 Normal=2
 WIND=> Weak=1 Strong=2
 TARGET CONCEPT:PLAY TENNIS=> Yes=10 No=5

Actual Values ['10', '5']
Predicted Values ['5', '5']
Accuracy :  50.0 %
"""