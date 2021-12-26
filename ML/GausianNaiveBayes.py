from ML.DataSet import DataSet
from ML.MLModel import MLModel
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold 

class GausianNaiveBayes(MLModel) :
    def __init__(self, dataset) :
        self.dataset = dataset
        X_train, X_test, y_train, y_test = train_test_split(dataset.GetXData(), dataset.GetYData(), test_size=0.5, random_state=0)
        gnb = GaussianNB()
        gnb.fit(X_train, y_train).predict(X_test)
        y_pred = gnb.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("matrix : ", confusion_matrix(y_test, y_pred))
        print("f1 score = ", f1_score(y_test, y_pred, zero_division=1))

    
    def Process(self, test_size = 0.5, random_state = 0) :
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.GetXData(), self.dataset.GetYData(), test_size = test_size, random_state = random_state)
        return self.__ProcessAlgorithm(X_train, y_train, X_test)

    def Process(self, x_train, y_train, x_test) :
        return self.__ProcessAlgorithm(x_train, y_train, x_test)
    
    def KFold(self, number) :
        print("K Fold")
        kf = KFold(n_splits = number, random_state = None)
        x_data = self.dataset.GetXData()
        y_data = self.dataset.GetYData()
        for train, test in kf.split(x_data) : 
            x_train, x_test = x_data.iloc[train, :], x_data.iloc[test,:]
            y_train, y_test = y_data.iloc[train], y_data.iloc[test]
            pred = self.Process(x_train, y_train, x_test)
            print(self.GetPrecision(y_test, pred))

    def GetConfusionMatrix(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)
    
    def GetF1Score(self, y_test, y_pred):
        return f1_score(y_test, y_pred, zero_division = 1)
    
    def GetAccurancy(self, y_test, y_pred):
        return metrics.accuracy_score(y_test, y_pred, zero_division = 1)

    def GetRecall(self, y_test, y_pred):
        return metrics.recall_score(y_test, y_pred, zero_division = 1)
    
    def GetPrecision(self, y_test, y_pred):
        return metrics.precision_score(y_test, y_pred, zero_division = 1)
    
    def GetF1Score(self, y_test, y_pred):
        return super().GetF1Score(y_test, y_pred)

    def GetPriorProbability(self) :
        return self.dataset.GetAmountOfClasses()

    
    def __ProcessAlgorithm(self, x_train, y_train, x_test) : 
        gnb = GaussianNB()
        ml_process = gnb.fit(x_train, y_train)
        return ml_process.predict(x_test)
