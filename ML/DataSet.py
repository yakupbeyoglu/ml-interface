import pandas
import os
class DataSet:
    
    def __init__(self, path, class_index) :
        if not os.path.exists(path) : 
            raise f'Given csv file with {path} is not exists'
        self.csv_path = path

        if class_index < 0 : 
            raise f'Given classes index {class_index} is not valid'
        self.class_index = class_index
        self.dataset = pandas.read_csv(self.csv_path)
        if class_index > len(self.dataset.columns) : 
            raise f'Given classes index is {class_index} but number of columns in data set is {self.dataset.columns}'
        self.X = self.dataset.iloc[:, 0 : class_index - 1]
        self.Y = self.dataset.iloc[:, class_index - 1]
        
    def GetSize(self):
        return len(self.dataset.row)
    
    def GetDataSet(self) :
        return self.dataset 
    
    def GetXData(self) : 
        return self.X
    
    def GetYData(self) :
        return self.Y
    
    def GetXYData(self):
        return self.X, self.Y
