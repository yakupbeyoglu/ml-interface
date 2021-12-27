from ML.DataSet import DataSet
from ML.MLModel import MLModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn import preprocessing

class LogisticRegressionModel(MLModel):
    def __init__(self, dataset):
        self.dataset = dataset
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.GetXData(), dataset.GetYData(), test_size=0.5, random_state=0)
        regression = LogisticRegression(max_iter=1000)
        regression.fit(X_train, y_train)
        pred = regression.predict(X_test)
        score = 0
        score = regression.score(X_test, y_test)
        print("my score = ", score)

    def Process(self, test_size=0.5, random_state=0):
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.GetXData(
        ), self.dataset.GetYData(), test_size=test_size, random_state=random_state)
        return self.__ProcessAlgorithm(X_train, y_train, X_test)

    def Process(self, x_train, y_train, x_test):
        return self.__ProcessAlgorithm(x_train, y_train, x_test)

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

    def KBest(self, number, numberoffeature = 4):
        print("K best-Fold")
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
            x_train_selected, x_test_selected = self.SelectFeatures(numberoffeature, x_train, y_train, x_test)
            pred = self.Process(x_train_selected, y_train, x_test_selected)
            predictionresult["precision"] += self.GetPrecision(y_test, pred)
            predictionresult["recall"] += self.GetRecall(y_test, pred)
            predictionresult["f1-score"] += self.GetF1Score(y_test, pred)
            predictionresult["accurancy"] += self.GetAccurancy(y_test, pred)

        for value in predictionresult:
            predictionresult[value] /= number
        return predictionresult

    def SelectFeatures (self, number, x_train, y_train, x_test):  
        featureselect = SelectKBest(score_func = f_regression, k = number)
        featureselect.fit(x_train, y_train)
        scores = []
        for score in featureselect.scores_ : 
            scores.append('{0:f}'.format(score))
        print("Scores = ", scores)
        x_train_selected = featureselect.transform(x_train)
        x_test_selected = featureselect.transform(x_test)
        cols = featureselect.get_support(indices=True)
        features_df_new = x_train.iloc[:,cols]
        print("selected features : \n ", features_df_new)
        return x_train_selected, x_test_selected
        

    def GetConfusionMatrix(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)

    def GetF1Score(self, y_test, y_pred):
        return f1_score(y_test, y_pred)

    def GetAccurancy(self, y_test, y_pred):
        return metrics.accuracy_score(y_test, y_pred)

    def GetRecall(self, y_test, y_pred):
        return metrics.recall_score(y_test, y_pred, zero_division=0)

    def GetPrecision(self, y_test, y_pred):
        return metrics.precision_score(y_test, y_pred, zero_division=0)

    def GetPriorProbability(self):
        return self.dataset.GetAmountOfClasses()

    def __ProcessAlgorithm(self, x_train, y_train, x_test):
        regression = LogisticRegression(max_iter=1000)
        ml_process = regression.fit(x_train, y_train)
        return ml_process.predict(x_test)
