#!/usr/bin/env python3
import argparse
import logging
import csv
from pathlib import Path
from ML.DataSet import DataSet
from ML.GausianNaiveBayes import GausianNaiveBayes
from ML.LogisticRegression import LogisticRegressionModel
from ML.Enums.ActivationFunctions import ActivationFunctions
from ML.Ann import Ann
from ML.Knn import Knn


argparser = argparse.ArgumentParser(
    description='Run this software with csv path and class index in the csv file'
)

argparser.add_argument('--csv', type=str, help='CSV file path')
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

result_path="data_size_500_resized_2"
modelname = "knn-k=1_n=5"
Path(result_path).mkdir(parents=True, exist_ok=True)
field_names = ['precision', 'recall', 'f1-score', 'accuracy']

knn = Knn(dataset, 1)
kfoldresult  = knn.KFold(5)
print("results are ready")
print(kfoldresult)
print(kfoldresult)
kfold_history= [kfoldresult]
with open(f'./{result_path}/{modelname}.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(kfold_history)
