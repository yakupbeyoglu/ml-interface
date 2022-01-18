from keras.models import Sequential
from keras.layers import Activation, Dense
from ML.ActivationFunctions import ActivationFunctions


class AnnModel:

    # initalize keras ann model
    def __init__(self, number_of_input):
        self.model = Sequential()
        self.number_of_input = number_of_input

    def AddLayer(self, number_of_nodes, activation="relu", input_dim=None):
        # check activation function is exist or not
        if not ActivationFunctions.IsExist(activation):
            return False

        if number_of_nodes <= 0:
            return False

        if not input_dim == None:
            self.model.add(Dense(number_of_nodes, input_dim=input_dim,
                           activation=ActivationFunctions.GetName(activation)))
        else:
            self.model.add(Dense(number_of_nodes,
                           activation=ActivationFunctions.GetName(activation)))

    def AddBinaryClassficationLayer(self, activation="relu"):
        # check activation function is exist or not
        if not ActivationFunctions.IsExist(activation):
            return False
        self.model.add(Dense(1, ActivationFunctions.GetName(activation)))
