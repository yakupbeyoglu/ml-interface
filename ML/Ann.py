from ML.Enums.ActivationFunctions import ActivationFunctions
from ML.MultiThread import MultiThread
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
import matplotlib.pyplot as pyplot


class Ann(MLModel):
    def __init__(self, dataset, epoch, batch_size, log_verbose=False):
        self.dataset = dataset
        self.epoch = epoch
        self.batch = batch_size
        self.log_status = log_verbose
        self.model = None
        self.ml_process_history = None
        self.prediction_model = None

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

    def Train(self):
        self.__CheckModel()
        x_train = self.dataset.GetXData()
        y_train  = self.dataset.GetYData()
        self.model.Compile()
        self.prediction_model = self.model.Fit(x_train, y_train, self.epoch, self.batch)

    def Predict(self, test_dataset):
        self.__CheckModel()
        assert not self.prediction_model == None, 'Ann model not trained!'
        x_test = test_dataset.GetDataSet()
        return self.model.MakeBinaryPredictions(x_test)

    def QuickProcess(self, test_size=0.2, random_state=0, validation_split_rate=None):
        self.__CheckModel()
        # ValidationFit
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.GetXData(
        ), self.dataset.GetYData(), test_size=test_size)
        self.pred_history, pred_result = self.__ProcessAlgorithm(
            X_train, y_train, X_test)
            
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
        return self.pred_history, predictionresult, pred_result

    def QuickProcessGraph(self, test_size=0.5, random_state=0):
        self.__CheckModel()
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.GetXData(
        ), self.dataset.GetYData(), test_size=test_size)
        self.pred_history, pred_result = self.__ProcessAlgorithmValidation(
            X_train, y_train, X_test, y_test)
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
        print("YES started")
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
        print(predicted)

    def __CheckModel(self):
        if self.model == None:
            assert("ML Model is not created, can not process without model")

    def __ProcessAlgorithm(self, x_train, y_train, x_test, validation_split=0):
        self.model.Compile()
        ml_process = self.model.Fit(x_train, y_train, self.epoch, self.batch)
        return ml_process, self.model.MakeBinaryPredictions(x_test)

    def __ProcessAlgorithmValidation(self, x_train, y_train, x_test, y_test):
        self.model.Compile()
        ml_process = self.model.Fit(
            x_train, y_train, self.epoch, self.batch, (x_test, y_test))
        return ml_process, self.model.MakeBinaryPredictions(x_test)

    def ExportModelAccuracyGraph(self, modeltitle, exportpath):
        pyplot.clf()

        if self.pred_history == None:
            raise("No history found, please Fit model before export graph")
        pyplot.plot(self.pred_history.history['accuracy'])
        pyplot.plot(self.pred_history.history['loss'])
        pyplot.title('Model  ' + modeltitle + ' Accuracy & Loss')
        pyplot.ylabel('accuracy')
        pyplot.xlabel('epoch')
        pyplot.legend(['Accuracy', 'Loss'], loc='upper left')
        pyplot.savefig(exportpath + modeltitle + '-Accuracy-Loss.png')
        # clear plot
        pyplot.clf()
        # history of loss
        
    



    def MultiThreadKFold(self, number):
        print("K Fold")
        kf = KFold(n_splits=number, random_state=None)
        x_data = self.dataset.GetXData()
        y_data = self.dataset.GetYData()

        predictionresult = {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "accuracy": 0
        }
        i = 1
        thread_array = []
        x_trains = []
        y_trains = []
        x_tests = []
        y_tests = []
        for train, test in kf.split(x_data):
            print(f'fold index = {i}')
            x_train, x_test = x_data.iloc[train, :], x_data.iloc[test, :]
            y_train, y_test = y_data.iloc[train], y_data.iloc[test]
            x_trains.append(x_train)
            y_trains.append(y_train)
            x_tests.append(x_test)
            y_tests.append(y_test)
            thread_array.append(MultiThread(target = self.Process, args=(x_train, y_train, x_test)))
            thread_array[-1].start()
        
        
        for fold in range(number):
            pred, y_pred = thread_array[fold].Join()
            predictionresult["precision"] += precision_score(
                y_tests[fold], y_pred, average="macro")
            predictionresult["recall"] += recall_score(
                y_tests[fold], y_pred)
            predictionresult["f1-score"] += f1_score(
                y_tests[fold], y_pred, average="macro")
            predictionresult["accuracy"] += accuracy_score(
                y_tests[fold], y_pred)
        
        for value in predictionresult:
            predictionresult[value] /= number
        return predictionresult


    def KFold(self, number):
        print("K Fold")
        kf = KFold(n_splits=number, random_state=None)
        x_data = self.dataset.GetXData()
        y_data = self.dataset.GetYData()

        predictionresult = {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "accuracy": 0
        }
        i = 1
        for train, test in kf.split(x_data):
            print(f'Fold {i}')
            x_train, x_test = x_data.iloc[train, :], x_data.iloc[test, :]
            y_train, y_test = y_data.iloc[train], y_data.iloc[test]
            pred, y_pred = self.Process(x_train, y_train, x_test)
            predictionresult["precision"] += precision_score(
                y_test, y_pred, average="macro")
            predictionresult["recall"] += recall_score(
                y_test, y_pred)
            predictionresult["f1-score"] += f1_score(
                y_test, y_pred, average="macro")
            predictionresult["accuracy"] += accuracy_score(
                y_test, y_pred)
            i += 1 

        # to get average
        for value in predictionresult:
            predictionresult[value] /= number
        return predictionresult
        
    def EvaluateModel(self, x_test, y_test):
        return self.model.EvaluateModel(x_test, y_test)
    
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
