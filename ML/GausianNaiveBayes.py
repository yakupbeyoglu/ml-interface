from ML.DataSet import DataSet
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class GausianNaiveBayes :
    def __init__(self, dataset) :
        self.dataset = dataset
        X_train, X_test, y_train, y_test = train_test_split(dataset.GetXData(), dataset.GetYData(), test_size=0.5, random_state=0)
        gnb = GaussianNB()
        gnb.fit(X_train, y_train).predict(X_test)
        y_pred = gnb.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("matrix : ", confusion_matrix(y_test, y_pred))
        print("f1 score = ", f1_score(y_test, y_pred, zero_division=1))
    def GetPriorProbability(self) :
        return self.dataset.GetAmountOfClasses()
    
