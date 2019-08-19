import csv
with open('trainingexamples.csv') as csvFile:
    data = [line[:-1] for line in csv.reader(csvFile) if line[-1] == "Y"]
    
for d in data:
    print(d)
print("")

S = ['@']*len(data[0])   # Initializing.
print("output in each steps are:\n{}".format(S))
for example in data:
    i = 0
    for feature in example:
        S[i] = feature if S[i] == '@' or S[i] == feature else '?'
        i += 1
    print(S)
