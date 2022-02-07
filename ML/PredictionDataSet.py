import pandas
import numpy as np
import os
import matplotlib.pyplot as plotter
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

class PredictionDataSet:
    
    def __init__(self, path) :
        if not os.path.exists(path) : 
            raise f'Given csv file with {path} is not exists'
        self.csv_path = path
        self.dataset = pandas.read_csv(self.csv_path)
        
        
    def GetSize(self):
        return len(self.dataset.row)
    
    def GetNumberOfColumn(self):
        return len(self.dataset.columns)
    
    def GetDataSet(self) :
        return self.dataset 
    
    def GetXData(self) : 
        return self.X
