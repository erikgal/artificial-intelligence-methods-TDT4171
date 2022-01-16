# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
from random import uniform


class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        self.hidden_weights = np.array([[-0.24585597, -0.35624455, -0.08753614, -0.38636473, -0.15514388,
                                         0.25016644, -0.39438316, -0.4850381, -0.36970219, 0.14245649,
                                         0.23851414, 0.14895151, -0.11309428, -0.37820334, 0.30995031,
                                         0.4395891, -0.46918064, 0.43099235, 0.41417165, 0.1105867,
                                         -0.00633143, -0.30645082, 0.454717, 0.26099371, 0.10541231,
                                         -0.23355796, -0.37927653, 0.48330808, -0.16026107, -0.07031951],
                                        [-0.39046186, 0.04367773, -0.37985042, -0.15715108, 0.44157353,
                                         -0.02915049, 0.34143821, -0.49712725, 0.47735484, 0.2429425,
                                         -0.2922108, -0.03311547, -0.19342243, 0.43407327, -0.0163172,
                                         0.23861905, -0.29968779, 0.28368037, -0.24054083, 0.45695404,
                                         0.06396844, -0.03097349, 0.26152171, 0.37151348, 0.46387169,
                                         0.34315667, -0.26355722, 0.08977726, -0.13640194, 0.31848837],
                                        [-0.42729355, 0.06161318, 0.40182782, 0.10980513, -0.34389647,
                                         -0.40375666, 0.00900577, -0.46533433, -0.27220363, -0.18860087,
                                         0.06512806, -0.23924442, 0.49154328, -0.3896322, 0.13907436,
                                         -0.05712262, 0.06760523, 0.02170273, -0.22866006, 0.3494111,
                                         -0.48561279, -0.14497593, 0.24282497, 0.35389351, -0.0329516,
                                         -0.04185398, 0.23994588, 0.18631958, 0.41295421, 0.10261553],
                                        [0.21910654, 0.30841623, -0.22686186, 0.23044604, -0.02559942,
                                         0.48455797, 0.10613603, -0.06215953, 0.10787527, 0.44659012,
                                         0.19772816, -0.49097093, 0.49496479, -0.06970553, -0.44434953,
                                         0.02306829, -0.13364029, 0.33865123, -0.44806371, 0.0444668,
                                         -0.27287923, 0.40588593, 0.01269137, 0.0307978, 0.09896176,
                                         0.46539401, -0.29933617, 0.28903144, -0.12805609, -0.34344429],
                                        [-0.32457079, -0.20924682, 0.31040527, 0.26636646, 0.03127596,
                                         0.46751942, 0.42766796, -0.43853775, -0.43200089, -0.49606427,
                                         0.35276058, -0.05002665, 0.21708012, -0.49873643, 0.34261495,
                                         0.08227424, -0.30245807, -0.32985585, -0.34444154, -0.47656632,
                                         0.2088498, 0.15446319, -0.22240282, 0.26322648, -0.32750733,
                                         0.42222717, 0.1221711, 0.07932995, -0.37500987, -0.12341437],
                                        [-0.48051097, -0.332187, 0.2234282, -0.33288782, -0.34670437,
                                         -0.15425635, -0.1346256, 0.30298061, -0.31148243, 0.20778375,
                                         -0.38459563, -0.01461667, -0.43005976, 0.32400599, 0.17323101,
                                         -0.24111548, 0.22421315, -0.201955, -0.29477751, 0.10884867,
                                         -0.41304608, 0.3468999, 0.05012311, 0.07104158, 0.49434874,
                                         0.30791937, -0.40966255, 0.26841562, 0.23184573, -0.4840798],
                                        [0.17568137, -0.49291328, 0.13575912, -0.13977202, -0.15976671,
                                         0.26255295, 0.43343284, 0.0023671, -0.20636687, -0.29877181,
                                         0.26142276, -0.41337182, 0.05034914, -0.23277784, -0.48571851,
                                         -0.31522696, -0.46417082, 0.4011623, -0.04849233, -0.06686157,
                                         -0.23649193, 0.23307, 0.36341989, -0.29445386, -0.02190881,
                                         -0.22498297, 0.12973808, 0.10058634, -0.08577381, 0.09988],
                                        [0.00919525, 0.37612253, -0.22017471, 0.33822469, 0.42334567,
                                         0.02352566, 0.33812283, 0.45799525, 0.25302875, 0.17932249,
                                         -0.28759753, 0.18014166, 0.13190603, 0.49653028, -0.13205368,
                                         0.44595448, 0.22355481, 0.20936633, 0.38452744, 0.42587964,
                                         -0.41931929, 0.10969721, -0.15753463, 0.07843583, -0.3630034,
                                         -0.31826391, -0.24304778, -0.18316063, -0.09491039, -0.11417825],
                                        [0.4867324, 0.13236375, -0.43805259, -0.22943944, -0.09233407,
                                         0.19865006, -0.35546574, 0.15546834, -0.26963297, 0.38981369,
                                         0.1777285, -0.45218904, 0.1432335, -0.32727903, 0.32497148,
                                         0.47393596, -0.38980826, -0.24826172, 0.29969139, 0.05911874,
                                         0.30729389, -0.10844582, 0.27854355, 0.28934496, 0.48495372,
                                         0.01396212, -0.37666492, -0.10183079, -0.38683041, -0.05441502],
                                        [0.49958925, -0.39874659, -0.25728548, -0.24294334, 0.25890326,
                                         -0.06842005, 0.04257606, -0.00118836, -0.06038872, -0.15871983,
                                         0.18805524, -0.06660309, -0.09697775, -0.481327, 0.41473745,
                                         -0.13193409, -0.08298115, 0.3135946, -0.4597558, -0.0927961,
                                         -0.30061031, -0.19729928, -0.00813417, 0.1141833, -0.38000207,
                                         0.27608599, -0.4121123, -0.0622048, -0.31145437, -0.34474559],
                                        [0.19317969, -0.25251477, 0.30893219, -0.12455644, 0.47740407,
                                         0.24108933, 0.21284662, -0.27247015, -0.24986908, 0.28229051,
                                         -0.45918856, 0.02899656, 0.49644798, 0.03684076, -0.08589715,
                                         -0.14206056, -0.29788173, -0.34517506, 0.41012314, -0.26385134,
                                         -0.16450283, -0.33282382, -0.01005024, -0.14771814, -0.41174499,
                                         -0.35277395, -0.16981558, -0.07178246, 0.46161918, 0.44340668],
                                        [-0.26998401, -0.2825196, 0.45725344, 0.23001986, 0.24465401,
                                         -0.08449139, -0.47684955, -0.24255317, -0.38411323, -0.32989377,
                                         -0.26492715, 0.10653851, 0.32171959, -0.47749504, 0.23866606,
                                         0.24255018, -0.49052345, -0.09520805, 0.11348279, -0.19861235,
                                         0.34556448, 0.0617419, -0.44790934, -0.06704971, 0.04734376,
                                         0.03839811, 0.32495848, 0.37346999, -0.44317315, 0.41786473],
                                        [-0.45504119, 0.00923002, -0.30824493, 0.13890443, -0.09329233,
                                         -0.08354416, 0.12650514, -0.41175364, 0.20954772, 0.2122503,
                                         0.42016651, -0.31378304, -0.11416282, -0.40115229, -0.12626687,
                                         -0.37105371, 0.03135459, 0.18096052, -0.28680932, 0.36814509,
                                         0.31579625, 0.05950359, -0.1796775, 0.43242226, -0.03362412,
                                         0.49171451, -0.00918504, 0.15676764, 0.47449513, -0.30184424],
                                        [-0.30504261, -0.17439974, 0.19588408, -0.32860002, 0.17129554,
                                         0.27214983, 0.13992266, -0.16173362, -0.11171355, -0.01289187,
                                         -0.43121669, 0.11525066, -0.45081388, 0.14062698, 0.11055915,
                                         -0.06430639, -0.11930054, 0.09109324, -0.21882559, -0.03724746,
                                         -0.22165352, -0.35067664, 0.29084777, -0.12135436, -0.24417717,
                                         -0.14162882, -0.17369839, -0.03724372, -0.49450271, -0.44238005],
                                        [-0.39513567, 0.39722007, 0.1497587, -0.09429911, 0.05849435,
                                         0.29448289, 0.25981077, 0.449602, 0.10177727, -0.31285485,
                                         0.3286975, 0.09130384, -0.48431315, 0.43255811, 0.22925853,
                                         0.42873264, -0.28389811, -0.37270365, -0.03194137, 0.23593106,
                                         0.19730224, 0.02925877, -0.35917762, -0.44828821, 0.12400689,
                                         -0.2948142, 0.3030717, -0.46504052, 0.12018083, -0.12081097],
                                        [0.00541619, -0.44374467, 0.19858362, -0.3267632, -0.08292613,
                                         -0.03285748, -0.44735707, 0.13713197, 0.01782136, 0.36495959,
                                         -0.33876485, 0.13146878, -0.42338875, -0.42594557, 0.47514041,
                                         0.064175, -0.49033395, -0.25921787, -0.26966108, 0.45037682,
                                         -0.14853556, 0.01524822, 0.49990279, -0.48600774, -0.37440919,
                                         0.37748931, -0.35311851, -0.3909551, 0.2533667, 0.42335159],
                                        [0.33521407, 0.40780035, -0.0150401, 0.01117944, -0.41512874,
                                         -0.14781417, -0.20933652, 0.28991633, 0.25470699, -0.38135742,
                                         0.48405493, 0.22391737, -0.05025263, -0.08442376, 0.37042874,
                                         -0.26646027, -0.0245245, 0.34412295, -0.45542577, 0.39498983,
                                         0.16092915, -0.01744358, -0.17129514, 0.37094012, 0.49232084,
                                         -0.20722786, -0.45428443, 0.17588125, -0.13621676, 0.40755323],
                                        [-0.08044528, -0.48918675, 0.04947445, 0.26703958, -0.27433853,
                                         0.01800656, 0.34810806, -0.47590886, -0.08883746, -0.2337013,
                                         -0.39918345, 0.35031409, 0.2198992, 0.14220858, -0.2237161,
                                         -0.27815809, -0.08208308, 0.21906423, -0.02592272, 0.23884813,
                                         0.01152444, 0.24406376, 0.1859636, -0.3122654, -0.03287385,
                                         0.18172133, 0.2977023, 0.18545456, -0.39646264, 0.40747484],
                                        [0.29435445, -0.03496598, 0.42129116, 0.04740099, -0.16192304,
                                         -0.13799474, -0.46521626, -0.34688532, -0.35169993, 0.42822187,
                                         -0.1393339, 0.3412729, 0.37469114, -0.33700889, -0.17861998,
                                         0.32929331, -0.32387175, 0.27703276, 0.42073704, 0.19460127,
                                         -0.12114823, 0.09972698, 0.11492519, -0.3499759, -0.33905418,
                                         -0.32739222, 0.38301295, 0.06695573, -0.08900382, 0.40549551],
                                        [-0.24393069, 0.48891272, -0.08504827, -0.42335366, 0.19785403,
                                         0.28981928, 0.35600675, 0.11937043, -0.36878073, 0.18949516,
                                         -0.20146262, -0.36497422, -0.31076152, -0.32927037, -0.12894691,
                                         -0.22120032, -0.15762832, 0.17062593, 0.02188911, 0.33250484,
                                         0.04624509, -0.15131287, 0.31492213, 0.43792842, -0.06397235,
                                         0.34613722, -0.14277777, 0.15109768, -0.3121744, -0.3438474],
                                        [0.37703117, 0.37457141, -0.12479453, -0.32728383, -0.41042558,
                                         0.33990321, 0.35917933, 0.21372321, -0.04385117, -0.45760125,
                                         -0.25015796, -0.11851493, -0.14786725, -0.46321817, -0.16184299,
                                         -0.39536083, -0.17298492, -0.23839688, 0.29981399, -0.49932596,
                                         0.1628082, 0.24217096, 0.02854506, 0.34072281, -0.28855972,
                                         -0.14489548, 0.34856622, 0.21463737, -0.13368851, 0.19994685],
                                        [-0.18058865, 0.2294901, 0.46778715, 0.31467289, 0.08501126,
                                         0.24404728, 0.47902454, -0.27897212, 0.43387333, 0.31448408,
                                         0.12826102, 0.11711877, -0.17614058, -0.36053361, -0.18871346,
                                         -0.4388993, 0.44086087, 0.48081608, 0.06879968, -0.41224525,
                                         -0.38926168, 0.17360282, 0.32977803, -0.05454289, -0.29502654,
                                         -0.40938358, -0.19364744, -0.43990571, 0.18596748, 0.04719398],
                                        [-0.18441973, 0.20776455, 0.08548723, 0.47490197, 0.03307334,
                                         -0.27961555, 0.33899898, 0.15328683, 0.33414231, 0.46706991,
                                         -0.10688784, -0.34090855, -0.00330787, 0.10721232, -0.4806671,
                                         -0.47177053, -0.39386103, -0.12539571, -0.22872698, 0.18554357,
                                         0.19829388, 0.29415167, 0.17535922, 0.39511629, 0.07749929,
                                         0.12885983, 0.21480074, -0.29091654, -0.34471072, 0.33048157],
                                        [0.47336159, 0.07228361, 0.43384154, -0.43281293, -0.00339861,
                                         -0.40356654, -0.44883553, 0.029632, -0.10076615, -0.08841347,
                                         0.34210648, -0.42857642, -0.04976248, 0.23845822, -0.34628796,
                                         -0.31884955, 0.41281481, -0.36854332, 0.02104603, -0.15708821,
                                         -0.29335027, 0.23921092, -0.16810545, -0.39211693, -0.24072125,
                                         -0.49824431, -0.47481671, -0.4955802, 0.24760181, -0.35106649],
                                        [0.20366769, -0.25545931, -0.01280296, 0.06109579, -0.49755611,
                                         0.06093761, -0.02368093, 0.46165814, 0.1430174, 0.15479419,
                                         -0.34567649, 0.29013504, 0.00882296, -0.11677981, -0.1576555,
                                         -0.38018701, 0.11083509, -0.48404934, 0.22677282, -0.0330266,
                                         -0.2035727, -0.43711037, -0.01863277, 0.19967108, 0.30253697,
                                         0.40418844, -0.44094822, 0.19920399, 0.17769146, 0.20620239]])

        self.outputWeights = np.array([[0.26826713, 0.0459042, -0.2977047, 0.38589329, -0.02097277,
                                        0.08951864, 0.03236571, -0.27283648, -0.02511779, -0.28538661,
                                        0.17197576, -0.08052136, 0.23251545, -0.46094975, -0.04434569,
                                        0.44325898, 0.31984086, -0.01563952, 0.00732693, 0.22615479,
                                        0.15979614, -0.10502385, -0.31832619, 0.44669527, 0.37242725]])

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.
        self.layers = list()
        self.weights = list()
        self.bias = list()

        # Setting up all weights, biases
        if hidden_layer:
            # hidden_weights[i][j] <- weight from inputNode_j to hiddenNode_i
            #hidden_weights = np.array([[uniform(-0.5, 0.5) for j in range(input_dim)] for i in range(self.hidden_units)])
            hidden_weights = self.hidden_weights
            self.weights.append(hidden_weights)

            #hidden_bias = np.array([uniform(-0.5, 0.5) for j in range(self.hidden_units)])
            hidden_bias = np.zeros((1, self.hidden_units))
            self.bias.append(hidden_bias)

            # output_weights[i][j] <- weight from hiddenNode_j to outputNode_i
            #output_weights = np.array([[uniform(-0.5, 0.5) for j in range(self.hidden_units)] for i in range(1)])
            output_weights= self.outputWeights
            self.weights.append(output_weights)

            #output_bias = np.array([uniform(-0.5, 0.5) for j in range(1)])
            output_bias = np.zeros((1, 1))
            self.bias.append(output_bias)
        else:
            # output_weights[i][j] <- weight from hiddenNode_j to outputNode_i
            output_weights = np.array([[uniform(-0.5, 0.5) for j in range(input_dim)] for i in range(1)])
            self.weights.append(output_weights)

            output_bias = np.array([uniform(-0.5, 0.5) for i in range(1)])
            self.bias.append(output_bias)

        # Setting up all layers with bias
        input_activations = np.zeros(input_dim)
        self.layers.append({'output': input_activations})

        if hidden_layer:
            hidden_activations = np.zeros(self.hidden_units)
            self.layers.append({'output': hidden_activations})

        output_activations = np.zeros(1)
        self.layers.append({'output': output_activations})

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # TODO: Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class
        # for iteration in range(self.epochs):
        for iteration in range(400):
            for x, y in zip(self.x_train, self.y_train):
                # Propagate the inputs forward to compute the outputs

                # Set initial input activation
                for i, input_val in enumerate(x):  # First layer
                    self.layers[0]['output'][i] = input_val

                # Propagate all activations
                for l in range(len(self.layers) - 1):  # Skip first layer
                    weights = self.weights[l]
                    activations = self.layers[l]['output']
                    propagated_activations = np.dot(weights, activations) # + self.bias[l]
                    self.layers[l + 1]['output'] = self.sigmoid(propagated_activations)

                # Propagate deltas backward from output layer to input layer
                # Initial output delta-error calculation
                self.layers[-1]['delta'] = self.der_sigmoid(self.layers[-1]['output']) * \
                                           (y - self.layers[-1]['output'])



                for l in reversed(range(1, len(self.layers) - 1)):  # Start from the back
                    g = self.der_sigmoid(self.layers[l]['output'])
                    w_ij = np.transpose(self.weights[l])
                    delta = self.layers[l + 1]['delta']

                    self.layers[l]['delta'] = g * (np.dot(w_ij, delta))

                    #delta_bias = self.der_sigmoid(1) * self.bias[l] * delta
                    #self.bias[l] += self.lr * 1 * delta_bias    # Does not influence previous neurons

                w1 = np.matrix(self.layers[0]['output'])
                d1 = np.transpose(np.matrix(self.layers[1]['delta']))
                test1 = self.weights[0] + self.lr * np.dot(d1, w1)

                w2 = np.matrix(self.layers[1]['output'])
                d2 = np.transpose(np.matrix(self.layers[2]['delta']))

                test2 = self.weights[1] + self.lr * np.dot(d2, w2)

                # Update every weight in network using deltas
                for l in range(len(self.layers) - 1):
                    self.weights[l] += self.lr * np.dot(np.transpose(np.matrix(self.layers[l+1]['delta'])),  np.matrix(self.layers[l]['output']))


    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # TODO: Implement the forward pass.

        # Set initial input activation
        for i, input_val in enumerate(x):  # First layer
            self.layers[0]['output'][i] = input_val

        # Propagate all activations
        for l in range(len(self.layers) - 1):  # Skip first layer
            weights = self.weights[l]
            activations = self.layers[l]['output']
            propagated_activations = np.dot(weights, activations) #+ self.bias[l]
            self.layers[l + 1]['output'] = self.sigmoid(propagated_activations)
        return self.layers[-1]['output'][0]

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def der_sigmoid(self, output: np.ndarray) -> np.ndarray:
        return output * (1 - output)


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        print(accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
