import csv
import math

data = []
actual_values = []
predicted_values = []

for row_list in csv.reader(open('ConceptLearning.csv')):
	data.append(row_list)

split_percentage = int(0.9*len(data))
train = data[:split_percentage]
test = data[split_percentage:]

print("\nOUTLOOK=> Sunny=1 Overcast=2 Rain=3\n TEMPERATURE=> Hot=1 Mild=2 Cool=3\n HUMIDITY=> High=1 Normal=2\n WIND=> Weak=1 Strong=2\n TARGET CONCEPT:PLAY TENNIS=> Yes=10 No=5\n")

p_y = 0
for i in range(len(train)):
	if train[i][-1]=='10':
		p_y +=1
p_n = len(train)-p_y

y = p_y/len(train)
n = p_n/len(train)

for t in test:
	prod_y = y
	prod_n = n

	for i in range(len(t)-1):
		cpy = 0
		cpn = 0
		for j in range(len(train)-1):
			if t[i]==train[j][i] and train[j][-1]=="10":
				cpy+=1
			if t[i]==train[j][i] and train[j][-1]=="5":
				cpn+=1

		prod_y *= cpy/p_y
		prod_n *= cpn/p_n

	if prod_y>prod_n:
		actual_values.append(t[-1])
		predicted_values.append("10")

	else:
		actual_values.append(t[-1])
		predicted_values.append("5")

print("Actual Values",actual_values)
print("Predicted Values",predicted_values)

accuracy = 0
for i in range(len(actual_values)):
	if actual_values[i]==predicted_values[i]:
		accuracy+=1

print("Accuracy : ",accuracy/len(actual_values)*100,"%")
