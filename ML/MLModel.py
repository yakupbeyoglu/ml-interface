from abc import ABC, abstractmethod

class MLModel(ABC) :
    @abstractmethod
    # Process Model
    def Process(self, x_train, y_train, x_test) :
        pass

    @abstractmethod
    # Fold dataset with given number
    def KFold(self,  number) : 
        pass

    @abstractmethod
    # Return confusion matrix of the predicted dataset, it can use for recall, f1-score etc calculation
    def GetConfusionMatrix(self) : 
        pass

    @abstractmethod
    # Get F1 score with given test and prediction data
    def GetF1Score(self, y_test, y_pred) :
        pass

    @abstractmethod
    # Get accurancy of the trained model
    def GetAccurancy(self, y_test, y_pred) : 
        pass
    
    @abstractmethod
    # Get Recall value of the trained model
    def GetRecall(self, y_test, y_pred) :
        pass

    @abstractmethod
    # Get Precesion value of trhe trained model
    def GetPrecision(self, y_test, y_pred) : 
        pass


    
