import math

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the neural network architecture




# input_size = 2
# hidden_size = 3
# output_size = 1

# # Initialize weights
# W1 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
# W2 = [0.7, 0.8, 0.9]

# # Define the input and target output
# X = [0.1, 0.2]
# y_true = [0.5]




# Manually initialize values
input_size = int(input("Enter the input size: "))
hidden_size = int(input("Enter the hidden size: "))
output_size = int(input("Enter the output size: "))

# Manually initialize weights
print("Enter the weights for W1:")
W1 = []
for i in range(input_size):
    row = [float(x) for x in input().split()]
    W1.append(row)

W2 = [float(x) for x in input("Enter the weights for W2: ").split()]

# Manually initialize input and target output
X = [float(x) for x in input("Enter the input values (separated by space): ").split()]
y_true = [float(x) for x in input("Enter the target output value: ").split()]

# Your code for forward pass, backward pass, and weight updates goes here

# Forward pass
h_input = [sum([X[i] * W1[i][j] for i in range(input_size)]) for j in range(hidden_size)]
h_output = [sigmoid(x) for x in h_input]  # Applying sigmoid function
y_pred = sum([h_output[j] * W2[j] for j in range(hidden_size)])

# Calculate the loss
loss = 0.5 * (y_pred - y_true[0]) ** 2

# Backward pass
dloss_dy_pred = y_pred - y_true[0]
dloss_dW2 = [dloss_dy_pred * h_output[j] for j in range(hidden_size)]
dloss_dh_output = [dloss_dy_pred * W2[j] for j in range(hidden_size)]
dh_output_dh_input = [sigmoid_derivative(x) for x in h_input]  # Applying derivative of sigmoid
dh_input_dW1 = [[X[i] for i in range(input_size)] for _ in range(hidden_size)]

# Compute dloss_dW1
dloss_dW1 = []
for j in range(hidden_size):
    dloss_dW1_row = []
    for i in range(input_size):
        dloss_dW1_row.append(dloss_dh_output[j] * dh_output_dh_input[j] * dh_input_dW1[j][i])
    dloss_dW1.append(dloss_dW1_row)

# Update weights
learning_rate = 0.1
for j in range(hidden_size):
    W2[j] -= learning_rate * dloss_dW2[j]
    for i in range(input_size):
        W1[i][j] -= learning_rate * dloss_dW1[j][i]

print("Updated W1:", W1)
print("Updated W2:", W2)
