#import math                It has been found that np.exp gives better results with the arrays than math.exp
                                                                           #Programming notes
#For hyperprameter maybe use linspace and make the tuning automatic by saving the parameters of the best accurate model
#save the weights in an external file so we can skip this in the next run (optional for user)
#Date (25/11/2024)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


#Here i will import the extra libraries (after asking GLA for the permission to use them)
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

#Importing data and splitting them
df = pd.read_csv('ce889_dataCollection(baha).csv') #Collecting data from the CSV file (ps: momken odam t5liha tt5ad men el user)
x=df.iloc[0:,:1] #this takes the coloumn i want ( remember [from which row : which row, from which coloumn : ....])
y=df.iloc[0:,1:2]
vx=df.iloc[0:,2:3]
vy=df.iloc[0:,3:4]

#creating a scaling function that is used to normalize the given values from the CSV
def scaling(column):
    return (column - column.min()) / (column.max() - column.min())

#handling outliers by removing them since in this game the user might have gone to extreme places (There is no need to it that much But if we are going to use it here we need to change the max and min values)
def detect_outliers(df, threshold=3):
    z_scores = np.abs(zscore(df))
    return (z_scores > threshold).any(axis=1)  #now we have a mask of the outliers in outr data set
#observing the difference and it has been found that if the thershold is more than 3 alot of data is removed and if it less than 3 no data is removed
print(f"Number of rows before removing outliers{df.shape[0]}")
df = df[~detect_outliers(df)]
print(f"Number of rows after removing outliers{df.shape[0]}")

                                #it could've been done that both input merged and decrease the lines of code but i used that way to debug easier
# Split data into training and testing sets
X_train, X_test,y_train, y_test, vx_train, vx_test, vy_train, vy_test = train_test_split(x,y,vx,vy, test_size=0.2, random_state=7)

# Print results
print("X1_train:\n", X_train.shape)
print("X1_test:\n", X_test.shape)
print("X2_train:\n", y_train.shape)
print("X2_test:\n", y_test.shape)
print("y1_train:\n", vx_train.shape)
print("y1_test:\n", vx_test.shape)
print("y2_train:\n", vy_train.shape)
print("y2_test:\n", vy_test.shape)
#Normalizing the inputs and predticted outputs
x_t=scaling(X_train).to_numpy()
x_v=scaling(X_test).to_numpy()
y_t=scaling(y_train).to_numpy()
y_v=scaling(y_test).to_numpy()
vx_t=scaling(vx_train).to_numpy()
vx_v=scaling(vx_test).to_numpy()
vy_t=scaling(vy_train).to_numpy()
vy_v=scaling(vy_test).to_numpy()

'''
# I will save the normalized values in another CSV file to use it in matlab and plot it whenever i want
df.iloc[0:, :1]=x
df.iloc[0:,1:2]=y
df.iloc[0:,2:3]=vx
df.iloc[0:,3:4]=vy
df.to_csv('Normalized.csv')

'''
# Combine inputs and outputs
#inputs = np.column_stack((x_t, y_t))  # Shape: (num_samples, 2)  #without Bias
inputs = np.column_stack((x_t, y_t, np.ones(x_t.shape[0])))  #With Bias
# #so shape[0] will be the number of samples that we will use in the for loop in each eppoch & shape[1] number of input neurons
actual_outputs = np.column_stack((vx_t, vy_t))
# inputs_val=np.column_stack((x_v,y_v)) # without Bias
inputs_val = np.column_stack((x_v, y_v, np.ones(x_v.shape[0])))  # With Bias
outputs_val=np.column_stack((vx_v,vy_v))


#Initialization of Variables

eta = 0.1  # Learning rate
mom = 0.2  # Momentum

#Size configuration to insure shape and multiplaction working smoothly
n_inputs = inputs.shape[1]
n_hidden_neurons = 15 #according to matylab trial and error
n_outputs = actual_outputs.shape[1] #changable to be len(actual_outputs) if we are using a hardcoded example

#Weights with random values
wsh = np.random.rand(n_inputs, n_hidden_neurons)  #Input to hidden layer weights
wsy = np.random.rand(n_hidden_neurons + 1, n_outputs)  # Hidden to output layer weights (+1) is for bais
#wsy = np.random.rand(n_hidden_neurons, n_outputs)  #without bais

#Initializing momentum updates to be used in the momuntom part of the eqn
prev_dwsh = np.zeros_like(wsh)
prev_dwsy = np.zeros_like(wsy)

# Class to create layers as objects (so we will have 2 layers hidden and output as 2 main objects)
class neuron:
    def __init__(self, inps, ws):
        self.inps = np.array(inps)
        self.ws = np.array(ws)
        self.vs = np.dot(self.inps, self.ws)  #Weighted sum by multiplaying inps by weights and then it will go to activation function
        # init some values to be filled later using the eqns
        self.hiddens = np.zeros(self.ws.shape[1])  #hidden neurons
        self.outputs = np.zeros(self.ws.shape[1])  #output neurons
        self.es = np.zeros(len(actual_outputs))  #error array
        self.gd_ys = np.zeros(self.ws.shape[1])  #gradient decent of output
        self.gd_hs = np.zeros(self.ws.shape[1])  #gradient decent of hiddden

    def activation(self, layer):
        #Sigmoid activation function      that if h is given it returns hidden layer (that is used so if i have more than 1 actv fn)
        if layer == 'h':
            self.hiddens = 1 / (1 + np.exp(-0.99*self.vs))
            return self.hiddens
        else:
            self.outputs = 1 / (1 + np.exp(-0.99*self.vs))
            return self.outputs
    #calculating error compared to actual outputs given from the CSV file
    def error_calc(self, actual_outputs):
        self.es = np.array(actual_outputs)  - self.outputs
        return self.es
#Back prob
    def gradient_y(self):
        #Output layer gradients
        self.gd_ys = eta * self.es * self.outputs * (1 - self.outputs)
        return self.gd_ys

    def gradient_h(self, delta_ys, wsy):
        #Hidden layer gradients
        #summation = np.dot(delta_ys, np.array(wsy).T) #without bias
        summation = np.dot(delta_ys, np.array(wsy[:-1]).T)
        self.gd_hs = eta * summation * self.hiddens * (1 - self.hiddens)
        return self.gd_hs
    #name of the function says it all
    def update_hidden_weights(self, eta, prev_dwsh, momentum):
        global wsh
        dw = eta * np.outer(self.inps, self.gd_hs) + momentum * prev_dwsh
        wsh += dw
        return dw

    def update_output_weights(self, hidden, eta, prev_dwsy, momentum):
        global wsy
        hidden_with_bias = np.append(hidden, 1) #comment this to remove bais ;)
        dw = eta * np.outer(hidden_with_bias, self.gd_ys) + momentum * prev_dwsy
        #dw = eta * np.outer(hidden, self.gd_ys) + momentum * prev_dwsy
        wsy += dw
        return dw

stopping_counter=0
#Early stopping function AKA stopping critiria which checks the current RMSE and the RMSE of the lookback epoch
#and if the dfference is so little (<thershold) or the RMSE increases it keeps count and if that happens 3 times it stops
#Count part it so insure it's not a local minma since it might decrease later
def stopping_cr(epochs,rmse_validation, threshold, lookback):
    global stopping_counter
    if epochs < lookback:
        return False
    dif=rmse_validation[epochs] - rmse_validation[epochs-lookback]
    if abs(dif) < threshold or dif>0:
        stopping_counter+=1
    else:
        stopping_counter=0
    if stopping_counter == 3:
        return True


#function that does all the sequence and it's placed in a function to be used in grid search
def train_neural_network(epochs, momentum, inputs, actual_outputs):
    #int some variables to be filled later as arrays for RMSE values throught the trainign
    rmse_training = np.zeros(epochs)
    rmse_validation = np.zeros(epochs)
    global prev_dwsh, prev_dwsy

    for epoch in range(epochs):
        total_error = 0  #Avg error for training
        for i in range(inputs.shape[0]):
            #1:Forward Pass Hidden Layer
            hidden_neuron = neuron(inputs[i], wsh)
            hidden_outputs = hidden_neuron.activation('h')

            #2:Output Layer
            output_neuron = neuron(np.append(hidden_outputs, 1), wsy) #with bais
            #output_neuron = neuron(hidden_outputs, wsy) #without bais
            output_values = output_neuron.activation('o')

            #3:Error calc
            error = output_neuron.error_calc(actual_outputs[i])
            sqr_error = np.square(error)
            total_error += np.mean(sqr_error)

            #4:gradent decent calc
            output_neuron.gradient_y()
            hidden_neuron.gradient_h(output_neuron.gd_ys, wsy)

            #5:Updating Weights
            prev_dwsh = hidden_neuron.update_hidden_weights(eta, prev_dwsh, momentum)
            prev_dwsy = output_neuron.update_output_weights(hidden_outputs, eta, prev_dwsy, momentum)
        #calculating the RMSE for the training for each epoch
        RMSE_t = np.sqrt(total_error / inputs.shape[0])
        rmse_training[epoch] = RMSE_t

        #Validation
        total_val_error = 0
        for i in range(inputs_val.shape[0]):
            hidden_outputs = 1 / (1 + np.exp(-np.dot(inputs_val[i], wsh)))
            final_outputs = 1 / (1 + np.exp(-np.dot(np.append(hidden_outputs, 1), wsy))) #with bais
            #final_outputs = 1 / (1 + np.exp(-np.dot(hidden_outputs, wsy))) without bais
            error = (final_outputs - outputs_val[i])
            total_val_error += np.mean(np.square(error))

        RMSE_v = np.sqrt(total_val_error / inputs_val.shape[0])
        rmse_validation[epoch] = RMSE_v


        print(f"Epoch {epoch + 1}/{epochs}")
        print("Training RMSE:", RMSE_t)
        print("Validation RMSE:", RMSE_v)
        print("----------------------xxxxxxx----------------------------")
        #let's check for early stop
        if stopping_cr(epoch,rmse_validation,0.0001,7):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        else:
            print(f"pass counter is {stopping_counter}")
    #printing the weight to take it to the neural net holder (to.csv can be used here)
    print(f"weights for hidden layer are{wsh}")
    print(f"weights for output layer are{wsy}")

    #Plotting RMSE
    plt.plot(rmse_training, label='Training RMSE')
    plt.plot(rmse_validation, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()


epochs = int(input("Enter the number of epochs for training: "))
train_neural_network(epochs, mom, inputs, actual_outputs)