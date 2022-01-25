from ML.Enums.ActivationFunctions import ActivationFunctions
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from ML.DataSet import DataSet
from ML.MLModel import MLModel
from ML.AnnModel import AnnModel
from keras.models import Sequential
from keras.layers import Dense
from ML.Enums.Losses.Probabilistic import Probabilistic
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from math import sqrt
from math import pi
from math import exp


class Ann(MLModel):
    def __init__(self, dataset, epoch, batch_size, log_verbose=False):
        self.dataset = dataset
        self.epoch = epoch
        self.batch = batch_size
        self.log_status = log_verbose
        self.model = None

    def AddLayer(self, number_of_nodes, activation=ActivationFunctions.relu, input_dim=None):
        if number_of_nodes == None:
            assert("Number of nodes not initalized")
        if activation == None:
            assert("Activation not initalized")
        if self.model == None:
            self.model = AnnModel()
            print(f'input dim = ', input_dim)

        self.model.AddLayer(number_of_nodes, activation, input_dim)

    def AddBinaryClassificationLayer(self, activation=ActivationFunctions.relu):
        self.model.AddBinaryClassificationLayer(activation)

    def QuickProcess(self, test_size=0.5, random_state=0):
        self.__CheckModel()
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.GetXData(
        ), self.dataset.GetYData(), test_size=test_size)
        self.pred_history, pred_result = self.__ProcessAlgorithm(X_train, y_train, X_test)
        predictionresult = {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "accuracy": 0
        }
        predictionresult["precision"] += precision_score(
            y_test, pred_result, average="macro")
        predictionresult["recall"] += recall_score(
            y_test, pred_result, average="macro")
        predictionresult["f1-score"] += f1_score(
            y_test, pred_result, average="macro")
        predictionresult["accuracy"] += accuracy_score(
            y_test, pred_result)
        return self.pred_history, predictionresult

    def Process(self, x_train, y_train, x_test):
        self.__CheckModel()
        return self.__ProcessAlgorithm(x_train, y_train, x_test)

    def PlotModel(self, full_file_path):
        self.model.PlotModel(full_file_path)

    def BuildModel(self):
        self.model.Compile()

    def BinaryPredict(self, prediction_data):
        size = len(prediction_data)
        if size == 0:
            assert("Prediction data is empty")
        predicted = self.model.MakeBinaryPredictions(prediction_data)
        numberoftrue = 0
        for i in range(size):
            list = prediction_data[i].tolist()
            binaryclass = self.dataset.y[i]
            print(
                f'{list} => {predicted[i]} (expected class is {binaryclass})')
            if predicted[i] == binaryclass:
                numberoftrue += 1
        accurancy = numberoftrue * 100 / size
        print(f'Acurrancy is  = {accurancy} %')

    def __CheckModel(self):
        if self.model == None:
            assert("ML Model is not created, can not process without model")

    def __ProcessAlgorithm(self, x_train, y_train, x_test):
        self.model.Compile()
        ml_process = self.model.Fit(x_train, y_train, self.epoch, self.batch)
        return ml_process, self.model.MakeBinaryPredictions(x_test)

    def KFold(self, number):
        print("K Fold")
        kf = KFold(n_splits=number, random_state=None)
        x_data = self.dataset.GetXData()
        y_data = self.dataset.GetYData()

        predictionresult = {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "accurancy": 0
        }

        for train, test in kf.split(x_data):
            x_train, x_test = x_data.iloc[train, :], x_data.iloc[test, :]
            y_train, y_test = y_data.iloc[train], y_data.iloc[test]
            pred, y_pred = self.Process(x_train, y_train, x_test)
            predictionresult["precision"] += precision_score(
                y_test, y_pred, average="macro")
            predictionresult["recall"] += recall_score(
                y_test, y_pred)
            predictionresult["f1-score"] += f1_score(
                y_test, y_pred, average="macro")
            predictionresult["accurancy"] += accuracy_score(
                y_test, y_pred)

        # to get average
        for value in predictionresult:
            predictionresult[value] /= number
        return predictionresult

    def GetConfusionMatrix(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)

    def GetF1Score(self, y_test, y_pred):
        return f1_score(y_test, y_pred, zero_division=0)

    def GetAccurancy(self, y_test, y_pred):
        return metrics.accuracy_score(y_test, y_pred)

    def GetRecall(self, y_test, y_pred):
        return metrics.recall_score(y_test, y_pred, zero_division=0)

    def GetPrecision(self, y_test, y_pred):
        return metrics.precision_score(y_test, y_pred, zero_division=0)

    def AttachModel(self, model):
        self.model = model
