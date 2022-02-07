#!/usr/bin/env python3
import argparse
import logging
import csv
from pathlib import Path

from pandas.core.frame import DataFrame
from ML.DataSet import DataSet
from ML.GausianNaiveBayes import GausianNaiveBayes
from ML.LogisticRegression import LogisticRegressionModel
from ML.Enums.ActivationFunctions import ActivationFunctions
from ML.Ann import Ann
from ML.Knn import Knn
import matplotlib.pyplot as pyplot
import pandas as pd 
import numpy as np 

def ExportPredictionResult(predictiondata, filepath):
    if filepath == None:
        raise("Path is empty")
    file = open(filepath, 'w')
    writer = csv.writer(file)
    for i in range(len(predictiondata)):
        writer.writerow(predictiondata[i])

def ExportGraph(names, scores, modelname, path):
    pyplot.clf()
    pyplot.title('Model  ' + modelname + ' F1 scores')
    for score in scores:
        pyplot.plot(score)
    pyplot.ylabel('f1 scores')
    pyplot.xlabel('epoch')
    pyplot.legend(['Accuracy', 'Loss'], loc='upper left')
    pyplot.savefig(path + path + '-Accuracy-Loss.png')
    # clear plot
    pyplot.clf()
    # history of loss
        
argparser = argparse.ArgumentParser(
    description='Run this software with csv path and class index in the csv file'
)

argparser.add_argument('--csv', type=str, help='CSV file path')
argparser.add_argument('--test', type=str, help='test_data path')

argparser.add_argument('--index', type=int,
                       help='Index of the class in the csv file')

arguments = argparser.parse_args()
if not arguments.csv:
    raise argparse.ArgumentTypeError('csv is not given')

if not arguments.index:
    raise argparse.ArgumentTypeError('--index is not given')

print(
    f'csv path : {arguments.csv} \nindex of the class in csv : {arguments.index}')


dataset = DataSet(arguments.csv, arguments.index, False, True)
dataset.GetBarChartOfClasses()
x, y = dataset.GetXYData()
print(x)
print("Hey you can find values below\n", dataset.GetAmountOfClasses())

# result_path="data_size_500_resized_2"
# modelname = "knn-k=1_n=5"
# Path(result_path).mkdir(parents=True, exist_ok=True)
# field_names = ['precision', 'recall', 'f1-score', 'accuracy']

# knn = Knn(dataset, 1)
# kfoldresult  = knn.KFold(5)
# print("results are ready")
# print(kfoldresult)
# print(kfoldresult)
# kfold_history= [kfoldresult]
# with open(f'./{result_path}/{modelname}.csv', 'w') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=field_names)
#     writer.writeheader()
#     writer.writerows(kfold_history)

modelname = "Test"
Path(modelname).mkdir(parents=True, exist_ok=True)
ann = Ann(dataset, 1, 50, False)
ann.AddLayer(14, ActivationFunctions.relu, dataset.GetNumberOfColumn() - 1)
ann.AddLayer(11, ActivationFunctions.relu, dataset.GetNumberOfColumn() - 1)
ann.AddLayer(11, ActivationFunctions.relu, dataset.GetNumberOfColumn() - 1)

ann.AddBinaryClassificationLayer(ActivationFunctions.sigmoid)
ann.BuildModel()

ann.PlotModel(
    f'./{modelname}/model.png')
#kfoldresult = ann.KFold(5)
# print(kfoldresult)
history, history_result, pred_result = ann.QuickProcess(0.2)
ExportPredictionResult(pred_result, f'./{modelname}/myprediction.csv')
field_names = ['precision', 'recall', 'f1-score', 'accuracy']

ann.ExportModelAccuracyGraph(modelname, f'./{modelname}/')
print("MY HISTORY")
print(history_result)
history_result = [history_result]
print(history_result)
with open(f'./{modelname}/scores.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(history_result)

kfoldresult  = ann.KFold(5)
print(kfoldresult)
kfold_history= [kfoldresult]
with open(f'./{modelname}/kfoldscores.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(kfold_history)
