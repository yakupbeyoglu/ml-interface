from ML.DataSet import DataSet
class GausianNaiveBayes :
    def __init__(self, dataset) :
        self.dataset = dataset
    
    def GetPriorProbability(self) :
        return self.dataset.GetAmountOfClasses()
    
