#import math                It has been found that np.exp gives better results with the arrays than math.exp
#Programming notes
#For hyperparameter maybe use linspace and make the tuning automatic by saving the parameters of the best accurate model
#save the weights in an external file so we can skip this in the next run (optional for user)
#Date (25/22/2024)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from itertools import product

# Importing data and splitting them
df = pd.read_csv('ce889_dataCollection.csv')  # Collecting data from the CSV file (ps: momken odam t5liha tt5ad men el user)
x = df.iloc[0:, :1]  # This takes the column I want
y = df.iloc[0:, 1:2]
vx = df.iloc[0:, 2:3]
vy = df.iloc[0:, 3:4]

# Scaling function
def scaling(column):
    return (column - column.min()) / (column.max() - column.min())

# Handling outliers by removing them since in this game the user might have gone to extreme places
def detect_outliers(df, threshold=3):
    z_scores = np.abs(zscore(df))
    return (z_scores > threshold).any(axis=1)  # Now we have a mask of the outliers in our dataset

print(f"Number of rows before removing outliers: {df.shape[0]}")
df = df[~detect_outliers(df)]  # Removing outliers
print(f"Number of rows after removing outliers: {df.shape[0]}")

# Preprocessing
print(f"Number of rows before cleaning: {df.shape[0]}")
df.dropna(inplace=True)
print(f"Number of rows after cleaning: {df.shape[0]}")

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test, vx_train, vx_test, vy_train, vy_test = train_test_split(
    x, y, vx, vy, test_size=0.2, random_state=7
)

# Print results
print("X1_train:\n", X_train.shape)
print("X1_test:\n", X_test.shape)
print("X2_train:\n", y_train.shape)
print("X2_test:\n", y_test.shape)
print("y1_train:\n", vx_train.shape)
print("y1_test:\n", vx_test.shape)
print("y2_train:\n", vy_train.shape)
print("y2_test:\n", vy_test.shape)

# Plotting training data and testing data to ensure that no major part is missing in either and the samples are well chosen
#fig, axs = plt.subplots(4, 2, figsize=(12, 16))  # Preparing a grid to place the figures in
#mpl.rcParams['agg.path.chunksize'] = 10000

# # Plot each array in its own subplot
# arrays = [X_train, X_test, y_train, y_test, vx_train, vx_test, vy_train, vy_test]
# titles = ['x_t', 'x_v', 'y_t', 'y_v', 'vx_t', 'vx_v', 'vy_t', 'vy_v']
# for ax, array, title in zip(axs.flat, arrays, titles):
#     ax.plot(array)
#     ax.set_title(title)
#     ax.grid(True)

# # Adjust spacing between subplots
# plt.tight_layout()
# plt.show()

# Normalizing the inputs and predicted outputs
x_t = scaling(X_train).to_numpy()
x_v = scaling(X_test).to_numpy()
y_t = scaling(y_train).to_numpy()
y_v = scaling(y_test).to_numpy()
vx_t = scaling(vx_train).to_numpy()
vx_v = scaling(vx_test).to_numpy()
vy_t = scaling(vy_train).to_numpy()
vy_v = scaling(vy_test).to_numpy()

# Combine inputs and outputs
inputs = np.column_stack((x_t, y_t))  # Shape: (num_samples, 2)
print(inputs)
print(inputs.shape)  # So shape[0] will be the number of samples that we will use in the for loop in each epoch & shape[1] number of input neurons
actual_outputs = np.column_stack((vx_t, vy_t))  # Shape: (num_samples, 2)
inputs_val = np.column_stack((x_v, y_v))
outputs_val = np.column_stack((vx_v, vy_v))
print(actual_outputs)
print(actual_outputs.shape)

# Now I will use libraries to autotune the hyperparameters using grid search for different values (this might take long)

# Creating the grid for the 3 hyperparameters I want to find the best values for
hyperparameters = {
    "learning_rate": np.linspace(0.01, 0.5, 10),  # Learning rate values between 0.01 and 0.5
    "momentum": np.linspace(0.1, 0.9, 5),  # Momentum values between 0.1 and 0.9
    "hidden_neurons": [10,12,13, 15,17, 20]  # Different numbers of neurons in the hidden layer
}

# Grid search function
def grid_search(grid, inputs, actual_outputs, inputs_val, outputs_val):
    param_name = list(grid.keys())
    param_values = list(grid.values())
    best_params = None
    best_score = float('inf')  # Start with a big number and go down as we find a better score

    # Iterate through all combinations of hyperparameters
    for values in product(*param_values):
        hyperparams = dict(zip(param_name, values))
        print(f"Testing hyperparameters: {hyperparams}")

        # Train the model with current hyperparameters
        score = train_fn(train_neural_network, inputs, actual_outputs, inputs_val, outputs_val, **hyperparams)
        print(f"Validation score: {score}")

        # Update the best parameters if the current score is better
        if score < best_score:
            best_score = score
            best_params = hyperparams

    return best_params, best_score

# Training function for each hyperparameter combination
def train_fn(train_neural_network, inputs, actual_outputs, inputs_val, outputs_val, learning_rate, momentum, hidden_neurons):
    # Initialize network parameters
    global wsh, wsy, eta, mom, n_hidden_neurons
    eta = learning_rate
    mom = momentum
    n_hidden_neurons = int(hidden_neurons)

    # Initialize weights and other parameters
    wsh = np.random.rand(inputs.shape[1], n_hidden_neurons)
    wsy = np.random.rand(n_hidden_neurons, actual_outputs.shape[1])

    # Train the neural network
    train_neural_network(epochs=40, momentum=mom, inputs=inputs, actual_outputs=actual_outputs)

    # Evaluate on validation set
    total_val_error = 0
    for i in range(inputs_val.shape[0]):
        # Forward pass for validation
        hidden_outputs = 1 / (1 + np.exp(-np.dot(inputs_val[i], wsh)))
        final_outputs = 1 / (1 + np.exp(-np.dot(hidden_outputs, wsy)))

        # Calculate error for both outputs
        error = np.mean(np.abs(final_outputs - outputs_val[i]))
        total_val_error += error

    return total_val_error / inputs_val.shape[0]  # Return average validation error

# Class for neurons
class neuron:
    def __init__(self, inps, ws):
        self.inps = np.array(inps)
        self.ws = np.array(ws)
        self.vs = np.dot(self.inps, self.ws)  # Weighted sum
        self.hiddens = np.zeros(self.ws.shape[1])  # Initialize hidden activations
        self.outputs = np.zeros(self.ws.shape[1])  # Initialize output activations
        self.es = np.zeros(self.ws.shape[1])  # Error initialization
        self.gd_ys = np.zeros(self.ws.shape[1])  # Output layer delta
        self.gd_hs = np.zeros(self.ws.shape[1])  # Hidden layer delta

    def activation(self, layer):
        if layer == 'h':
            self.hiddens = 1 / (1 + np.exp(-eta * self.vs))
            return self.hiddens
        else:
            self.outputs = 1 / (1 + np.exp(-eta * self.vs))
            return self.outputs

    def error_calc(self, actual_outputs):
        self.es = np.array(actual_outputs) - self.outputs
        return self.es

    def gradient_y(self):
        self.gd_ys = eta * self.es * self.outputs * (1 - self.outputs)
        return self.gd_ys

    def gradient_h(self, delta_ys, wsy):
        summation = np.dot(delta_ys, np.array(wsy).T)
        self.gd_hs = eta * summation * self.hiddens * (1 - self.hiddens)
        return self.gd_hs

    def update_hidden_weights(self, eta, prev_dwsh, momentum):
        global wsh
        dw = eta * np.outer(self.inps, self.gd_hs) + momentum * prev_dwsh
        wsh += dw
        return dw

    def update_output_weights(self, hidden, eta, prev_dwsy, momentum):
        global wsy
        dw = eta * np.outer(hidden, self.gd_ys) + momentum * prev_dwsy
        wsy += dw
        return dw

# Training function
def train_neural_network(epochs, momentum, inputs, actual_outputs):
    global prev_dwsh, prev_dwsy
    prev_dwsh = np.zeros_like(wsh)
    prev_dwsy = np.zeros_like(wsy)

    for epoch in range(epochs):
        total_error = 0
        for i in range(inputs.shape[0]):
            # Forward pass
            hidden_neuron = neuron(inputs[i], wsh)
            hidden_outputs = hidden_neuron.activation('h')

            output_neuron = neuron(hidden_outputs, wsy)
            output_values = output_neuron.activation('o')

            # Error calculation
            error = output_neuron.error_calc(actual_outputs[i])
            total_error += np.mean(np.abs(error))

            # Backward pass
            output_neuron.gradient_y()
            hidden_neuron.gradient_h(output_neuron.gd_ys, wsy)

            # Update weights
            prev_dwsh = hidden_neuron.update_hidden_weights(eta, prev_dwsh, momentum)
            prev_dwsy = output_neuron.update_output_weights(hidden_outputs, eta, prev_dwsy, momentum)

        print(f"Epoch {epoch + 1}/{epochs} | Total Error: {total_error / inputs.shape[0]}")

# Run grid search
best_params, best_score = grid_search(hyperparameters, inputs, actual_outputs, inputs_val, outputs_val)
print("Best Hyperparameters:", best_params)
print("Best Validation Score:", best_score)

# Run training with the best parameters
eta = best_params['learning_rate']
mom = best_params['momentum']
n_hidden_neurons = int(best_params['hidden_neurons'])
#train_neural_network(epochs=50, momentum=mom, inputs=inputs, actual_outputs=actual_outputs)
