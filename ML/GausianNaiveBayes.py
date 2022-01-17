from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from ML.DataSet import DataSet
from ML.MLModel import MLModel
from math import sqrt
from math import pi
from math import exp


class GausianNaiveBayes(MLModel):
    def __init__(self, dataset):
        self.dataset = dataset

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

    def GetPriorProbability(self, normalize=True):
        return self.dataset.GetAmountOfClasses(normalize)

    def __Mean(self, key):
        mean = sum(self.dataset.GetXData()[key]) / \
            float(len(self.dataset.GetXData()[key]))
        print(f'Mean of {key} = {mean}')
        return mean

   #  Below For Probability calculation functions
    def __MeanArray(self, arr):
        mean = sum(arr) / float(len(arr))
        return mean

    def __StdDevArray(self, arr):
        avarage = self.__MeanArray(arr)
        variances = sum([(x-avarage)**2 for x in arr]) / \
            float(len(arr)-1)
        stddev = sqrt(variances)
        return stddev

    def __StdDev(self, key):
        avarage = self.__Mean(key)
        variances = sum([(x-avarage)**2 for x in self.dataset.GetXData()[key]]) / \
            float(len(self.dataset.GetXData()[key])-1)
        stddev = sqrt(variances)
        print(f'stddev of {key} = {stddev}')
        return stddev

    def __GetProbability(self, x, mean, stddev):
        exponent = exp(-((x-mean)**2 / (2 * stddev**2)))
        return (1 / (sqrt(2 * pi) * stddev)) * exponent

    def GetProbabilityOfColumn(self, columnname, value, zeros, ones):
            return self.__GetProbability(value, zeros[columnname]["mean"], zeros[columnname]["std_dev"])
            

    # Summaraze data based by classes
    def DataSetSummarize(self):
        summarize=[]
        for column in self.dataset.GetXData():
            data={
                "name": column,
                "mean": self.__Mean(column),
                "std_dev": self.__StdDev(column),
                "size": len(self.dataset.GetXData()[column])
            }
            summarize.append(data)
        return summarize

    # data set summarize based on class, column mean of class
    def DataSetSummarize(self, zeros=[], ones=[]):

        summarize_zeros = dict()
        summarize_ones = dict()
        for column in zeros:
            data = {
                "mean": self.__MeanArray(zeros[column]),
                "std_dev": self.__StdDevArray(zeros[column]),
                "size": len(zeros[column])
            }
            summarize_zeros[column] = data

        for column in ones:
            data = {
                "mean": self.__MeanArray(ones[column]),
                "std_dev": self.__StdDevArray(ones[column]),
                "size": len(ones[column])
            }
            summarize_ones[column] = data

        return summarize_zeros, summarize_ones

    # Seperate all items by class
    def separate_by_class(self):
        separated = dict()
        data = self.dataset.GetXData()
        labels = self.dataset.GetYData()
        zeros = dict()
        ones = dict()
        for column in data:
            if(column not in zeros.keys()):
                zeros[column] = []
            if(column not in ones.keys()):
                ones[column] = []
            set = data[column]
            for i in range(len(set)):
                if(labels[i] == 0):
                    zeros[column].append(set[i])
                else:
                    ones[column].append(set[i])
        return zeros, ones

    def summarize_by_class(self):
        separated = self.separate_by_class(self.dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.DataSetSummarize(rows)
        return summaries
