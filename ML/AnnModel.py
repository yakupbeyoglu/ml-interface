from keras.models import Sequential
from keras.layers import Activation, Dense
from ML.Enums.ActivationFunctions import ActivationFunctions
from ML.Enums.Losses.Probabilistic import Probabilistic
from keras.utils.vis_utils import plot_model
from pandas import DataFrame


class AnnModel:

    # initalize keras ann model
    def __init__(self):
        self.__model = Sequential()
        # ready for prediction
        self.__isready = False
        # check is compiled or not
        self.__iscompiled = False

    def AddLayer(self, number_of_nodes, activation="relu", input=None):
        # check activation function is exist or not
        if not ActivationFunctions.IsExist(ActivationFunctions, activation):
            return False

        if number_of_nodes <= 0:
            return False

        if not input == None:
            self.__model.add(Dense(number_of_nodes, input_dim=input,
                                   activation=ActivationFunctions.GetName(ActivationFunctions, activation)))
        else:
            self.__model.add(Dense(number_of_nodes,
                                   activation=ActivationFunctions.GetName(ActivationFunctions, activation)))

    def AddBinaryClassficationLayer(self, activation="relu"):
        # check activation function is exist or not
        if not ActivationFunctions.IsExist(activation):
            return False
        self.__model.add(Dense(1, ActivationFunctions.GetName(activation)))

    def Compile(self, loss_function=Probabilistic.binary_crossentropy, metrics=['accuracy']):
        loss_is_exsits = Probabilistic.IsExist(Probabilistic, loss_function)
        if not loss_is_exsits:
            AssertionError("Loss function is not exist")
        self.__model.compile(loss=Probabilistic.GetName(
            Probabilistic, loss_function), optimizer="adam", metrics=['accuracy'])
        self.__iscompiled = True

    def Fit(self, x_data, y_data, epochs, batch_size):
        if self.__iscompiled == False:
            assert("Model is not compiled, please compile and re-run!")
        self.__isready = True
        return self.__model.fit(x_data, y_data, epochs, batch_size)

    def MakeBinaryPredictions(self, predictions_data):
        if predictions_data == None:
            assert("Prediction data can not be empty !")
        if not self.__isready:
            assert("Model is not fitted, please compile & fit!")
        # round the value for prediction
        return (self.__model.predict(predictions_data) > 0.5).asType(int)

    def PlotModel(self, full_file_path="model_plot.png"):
        if not self.__iscompiled:
            assert("Model is not compiled yet")
        plot_model(self.__model, full_file_path, show_shapes=True,
                   show_layer_names=True, show_layer_activations=True)
