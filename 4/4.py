import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([.92], [.86], [.89]), dtype=float)
X = X / np.amax(X, axis=0)  # maximum of X array longitudinally

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


# Variable initialization
epoch = 5000  # Setting training iterations
lr = 0.1  # Setting learning rate

# weight and bias initialization
wh = np.random.uniform(size=(2, 3))
bh = np.random.uniform(size=(1, 3))
wout = np.random.uniform(size=(3, 1))
bout = np.random.uniform(size=(1, 1))

# draws a random range of numbers uniformly of dim x*y
for i in range(epoch):
    # Forward Propogation
    act_1 = sigmoid(np.dot(X, wh) + bh)  # ( (3*2).(2*3) ) + (1*3) ==>> (3*3)
    act_2 = sigmoid(np.dot(act_1, wout) + bout)  # ( (3*3).(3,1) ) + (1*1) ==>> (3*1)

    # Backpropagation Error * derivatives_sigmoid
    d2 = (y - act_2) * derivatives_sigmoid(act_2)  # ((3*1)-(3*1)).(3*1)   ==>> (3*1)
    d1 = (d2.dot(wout.T)) * derivatives_sigmoid(act_1)  # ( (3*1).(1*3) ).(3*3) ==>> (3*3)

    # dot product of next layer error and current layer op
    wh += X.T.dot(d1) * lr
    wout += act_1.T.dot(d2) * lr

print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", act_2)


"""

Input: 
[[0.66666667 1.        ]
 [0.33333333 0.55555556]
 [1.         0.66666667]]
Actual Output: 
[[0.92]
 [0.86]
 [0.89]]
Predicted Output: 
 [[0.89415189]
 [0.88038569]
 [0.89536088]]

"""