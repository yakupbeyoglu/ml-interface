import pandas
import numpy as np
import os
import matplotlib.pyplot as plotter
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

class DataSet:
    
    def __init__(self, path, class_index, normalize = True, isshuffle = False) :
        if not os.path.exists(path) : 
            raise f'Given csv file with {path} is not exists'
        self.csv_path = path

        if class_index < 0 : 
            raise f'Given classes index {class_index} is not valid'
        self.class_index = class_index
        self.dataset = pandas.read_csv(self.csv_path, header=None)
        if normalize: 
            scaler = MinMaxScaler()
            self.dataset = pandas.DataFrame(scaler.fit_transform(self.dataset), columns=self.dataset.columns, index=self.dataset.index)
        
        if class_index > len(self.dataset.columns) : 
            raise f'Given classes index is {class_index} but number of columns in data set is {self.dataset.columns}'
        self.X = self.dataset.iloc[:, 0 : class_index - 1]
        self.Y = self.dataset.iloc[:, class_index - 1]
        if isshuffle:
            self.X, self.Y = shuffle(self.X, self.Y)
        
    def GetSize(self):
        return len(self.dataset.row)
    
    def GetNumberOfColumn(self):
        return len(self.dataset.columns)
    
    def GetDataSet(self) :
        return self.dataset 
    
    def GetXData(self) : 
        return self.X
    
    def GetYData(self) :
        return self.Y
    
    def GetXYData(self) :
        return self.X, self.Y
    
    def GetNumberOfClasses(self) :
        return len(self.Y.value_counts())

    def GetAmountOfClasses (self, normalize = True) :
        map   = self.Y.value_counts(normalize)
        return map
    
    def PrintAmountOfClasses (self, normalize = True) :
        map   = self.GetAmountOfClasses(normalize)
        print(f'number of classes = {len(map)}')
        for key, value in map.iteritems() : 
            print (key, value)
    
    def GetBarChartOfClasses(self) :
        figure = plotter.figure()
        classes = []
        values = []
        map = self.GetAmountOfClasses(False)
        for key, value in map.iteritems() :
            classes.append(key)
            values.append(value)
        dataframe = pandas.DataFrame(values, index = classes)
        plot = pandas.concat([dataframe], axis=1).plot.bar(rot=0)
        plot.get_figure().savefig('output.pdf', format='pdf')
