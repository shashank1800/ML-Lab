import csv

with open('trainingexamples.csv') as csvFile:
    data = [line[:-1] for line in csv.reader(csvFile) if line[-1] == "Y"]

S = data[0]
print("POSITIVE EXAMPLES ARE: ",data)
print("Output in each steps are: \n")
print(['@']*len(S))
print(S)

for row in data:
    if row==S:
        continue
    i = 0
    for val in row:
        if S[i]!=val:
        	S[i] = '?'
        i += 1
    print(S)
