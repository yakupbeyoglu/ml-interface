from keras.models import Sequential
from keras.layers import Activation, Dense
from ML.Enums.ActivationFunctions import ActivationFunctions
from ML.Enums.Losses.Probabilistic import Probabilistic

class AnnModel:

    # initalize keras ann model
    def __init__(self):
        self.__model = Sequential()

    def AddLayer(self, number_of_nodes, activation="relu", input_dim=None):
        # check activation function is exist or not
        if not ActivationFunctions.IsExist(ActivationFunctions, activation):
            return False

        if number_of_nodes <= 0:
            return False

        if not input_dim == None:
            self.__model.add(Dense(number_of_nodes, input_dim=input_dim,
                           activation=ActivationFunctions.GetName(ActivationFunctions, activation)))
        else:
            self.__model.add(Dense(number_of_nodes,
                           activation=ActivationFunctions.GetName(ActivationFunctions, activation)))

    def AddBinaryClassficationLayer(self, activation="relu"):
        # check activation function is exist or not
        if not ActivationFunctions.IsExist(activation):
            return False
        self.__model.add(Dense(1, ActivationFunctions.GetName(activation)))
    
    def Compile(self, loss_function, metrics = ['accuracy']):
        loss_is_exsits  = Probabilistic.IsExist(Probabilistic, loss_function)
        if not loss_is_exsits:
            AssertionError("Loss function is not exist")
        self.__model.compile(loss=Probabilistic.GetName(Probabilistic, loss_function), optimizer="adam", metrics=['accuracy'])
    
    def Fit(self, x_data, y_data, epochs, batch_size):
        accurancy = self.__model.evaluate(x_data, y_data)
        print(f'Accuracy : {accurancy}')
