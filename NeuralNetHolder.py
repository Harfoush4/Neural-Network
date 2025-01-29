import numpy as np

class neuron:
    def __init__(self, inps, ws):
        self.inps = np.array(inps)
        self.ws = np.array(ws)
        self.vs = np.dot(self.inps, self.ws)  # Weighted sum
        self.hiddens = np.zeros(self.ws.shape[1])  # Initialize hidden activations
        self.outputs = np.zeros(self.ws.shape[1])  # Initialize output activations
        #self.es = np.zeros(len(actual_outputs))  # Error initialization
        #self.gd_ys = np.zeros(self.ws.shape[1])  # Output layer delta
        #self.gd_hs = np.zeros(self.ws.shape[1])  # Hidden layer delta

    def activation(self, layer):
        # Sigmoid activation function
        if layer == 'h':
            self.hiddens = 1 / (1 + np.exp(-0.9*self.vs))
            return self.hiddens
        else:
            self.outputs = 1 / (1 + np.exp(-0.9*self.vs))
            return self.outputs


class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.weightsh=[[ 1.03836913,  2.36806616 , 1.27618052, -1.39276823 , 6.69020476, -1.07442192,
   8.21462095 ,-0.07544661],
 [ 0.4515546 ,  1.32233404 , 1.35507259 , 1.02626652, -0.41703071 , 0.62394654,
  -0.19060612 , 5.58428387],
 [ 0.46187807 , 0.04951344 , 0.37191654 , 0.99390402, -3.01751414,  0.67940092,
  -3.27905539 , 0.25740749]]
 #            [[-1.76289924, 0.47683981 , 1.13026613 , 0.40621891 , 1.29346852 , 4.4586973,
 #   0.17806224 , 1.12525304],
 # [ 1.54060515 , 4.9895775,  -0.1954079,   0.69843271 ,-0.47013742, -0.0583763,
 #  -2.88180486 ,-0.08822543]]




        self.weightsy=[[  0.01372197 , -1.77085417],
 [  0.74317918,  -6.91405856],
 [ -1.16739402 , -2.97859069],
 [  2.25162679 ,  6.35944113],
 [-11.16891962 ,  6.85191718],
 [  2.36269928 ,  4.65287534],
 [ 11.39313472 , -0.32452071],
 [ -9.36612209 ,  0.88832898],
 [  4.57763116  ,-1.6717935 ]]
        #[[ 6.14152516, -2.64383391],
 # [-9.92784205, -2.21761442],
 # [-3.08097577 , 3.12050426],
 # [ 0.01624278 , 0.84858068],
 # [-3.80087766 , 3.39710256],
 # [12.62590236 ,-2.40261975],
 # [ 2.7644295  ,-4.75913475],
 # [-2.57250921,  3.1959549 ]]








    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        input_row = list(map(float, input_row.split(','))) #taking the CSV row and making it a list

        max_x=640.113
        max_y=650.402
        max_vx=7.999
        max_vy=7.864
        min_x=-641.203
        min_y=66.1015
        min_vx=-6.728294
        min_vy=-7.920305
        #Normalizing input by the same way the neural network was trained
        input_row[0]=(input_row[0]-min_x)/(max_x-min_x)
        input_row[1]=(input_row[1]-min_y)/(max_y-min_y)
        #adding the bais
        input_row.append(1)

        #Forward Pass
        hidden_neuron = neuron(input_row, self.weightsh)
        hidden_outputs = list(hidden_neuron.activation('h'))
        hidden_outputs.append(1) #adding the second bais

        output_neuron = neuron(hidden_outputs, self.weightsy)
        output_values = output_neuron.activation('o')

        #Step4 Denormaliz:
        vx=output_values[0]*(max_vx-min_vx)+min_vx
        vy=output_values[1]*(max_vy-min_vy)+min_vy
        denormalized_output = [vx,vy]


        return denormalized_output

        pass
