from math import sqrt
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
from math import sqrt
from math import pi
from math import exp


class Ann(MLModel):
    def __init__(self, dataset, epoch, batch_size, log_verbose = False):
        self.dataset = dataset
        self.epoch = epoch
        self.batch = batch_size
        self.log_status = log_verbose

    def AddLayer(self, number_of_nodes, activation="relu", input_dim=None):
        self.model = AnnModel()
        self.model.AddLayer(number_of_nodes, input_dim, activation)
        print(self.model)
    def Process(self, test_size=0.5, random_state=0):
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.GetXData(
        ), self.dataset.GetYData(), test_size=test_size, random_state=random_state)
        return self.__ProcessAlgorithm(X_train, y_train, X_test)

    def Process(self, x_train, y_train, x_test):
        return self.__ProcessAlgorithm(x_train, y_train, x_test)

    def __ProcessAlgorithm(self, x_train, y_train, x_test):
        gnb = GaussianNB()
        ml_process = gnb.fit(x_train, y_train)
        return ml_process.predict(x_test)

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
            pred = self.Process(x_train, y_train, x_test)
            predictionresult["precision"] += self.GetPrecision(y_test, pred)
            predictionresult["recall"] += self.GetRecall(y_test, pred)
            predictionresult["f1-score"] += self.GetF1Score(y_test, pred)
            predictionresult["accurancy"] += self.GetAccurancy(y_test, pred)

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

    def AttachModel(self, model) : 
        self.model = model
