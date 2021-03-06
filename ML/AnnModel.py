import numpy as np
np.random.seed(2017)
import random as rn
rn.seed(1000)
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'
import tensorflow as tf
tf.random.set_seed(123)

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

    def EvaluateModel(self, x_test, y_test):
        return self.__model.evaluate(x_test, y_test)
        
    def AddBinaryClassificationLayer(self, activation="relu"):
        # check activation function is exist or not
        if not ActivationFunctions.IsExist(ActivationFunctions, activation):
            return False
        self.__model.add(Dense(1, ActivationFunctions.GetName(
            ActivationFunctions, activation)))

    def Compile(self, loss_function=Probabilistic.binary_crossentropy, metrics=['accuracy']):
        loss_is_exsits = Probabilistic.IsExist(Probabilistic, loss_function)
        if not loss_is_exsits:
            AssertionError("Loss function is not exist")
        self.__model.compile(loss=Probabilistic.GetName(
            Probabilistic, loss_function), optimizer="adam", metrics=['accuracy'])
        self.__iscompiled = True

    def Fit(self, x_data, y_data, epochs, batch_size, validation_split_rate=0):
        if self.__iscompiled == False:
            assert("Model is not compiled, please compile and re-run!")
        self.__isready = True
        return self.__model.fit(x_data, y_data, batch_size, epochs, shuffle=False)

    def MakeBinaryPredictions(self, predictions_data):
        if not self.__isready:
            assert("Model is not fitted, please compile & fit!")
        # round the value for prediction
        return (self.__model.predict(predictions_data) > 0.5).astype(int)

    def PlotModel(self, full_file_path="model_plot.png"):
        if not self.__iscompiled:
            assert("Model is not compiled yet")
        plot_model(self.__model, full_file_path, show_shapes=True,
                   show_layer_names=True)
