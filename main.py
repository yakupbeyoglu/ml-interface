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

dataset = DataSet(arguments.csv, arguments.index, True, False)
dataset.GetBarChartOfClasses()
x, y = dataset.GetXYData()
print(x)
print("Hey you can find values below\n", dataset.GetAmountOfClasses())
'''
# Question 1 
gausian_naive_bayes = GausianNaiveBayes(dataset)
print("prior probability-normalized  : \n", gausian_naive_bayes.GetPriorProbability())
print("prior probability-not-normalized  : \n", gausian_naive_bayes.GetPriorProbability(False))
print("below : \n\n\n")
# Question 2
print(gausian_naive_bayes.DataSetSummarize())
zeros, ones = gausian_naive_bayes.separate_by_class()
zeros, ones = gausian_naive_bayes.DataSetSummarize(zeros,ones)
gausian_naive_bayes.GetProbabilityOfColumn("age", 60, zeros,ones)
gausian_naive_bayes.GetProbabilityOfColumn("anaemia", 1, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("creatinine_phosphokinase", 315, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("diabetes", 1, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("ejection_fraction", 60, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("high_blood_pressure", 0, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("platelets", 454000, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("serum_creatinine", 1.1, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("serum_sodium", 131, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("sex", 1, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("smoking", 1, zeros, ones)
gausian_naive_bayes.GetProbabilityOfColumn("time", 10, zeros, ones)
#Question 3 
values = gausian_naive_bayes.KFold(5)
print("Question 4 : Naive bayes KFOLD 5 :\n")
for i in values : 
    print(f'\t {i} = {values[i]}')
# Logsitict regression Kfold(5)
regression = LogisticRegressionModel(dataset)
values = regression.KFold(5)
print("Question 5: Logistic Regression KFOLD 5 :\n")
for i in values : 
    print(f'\t {i} = {values[i]}')
values = regression.KBest(5, 5)
print("Quetion 7 : Best Feature = \n")
for i in values : 
    print(f'\t {i} = {values[i]}')
'''

## Research Answers
modelname = "12-node-3-hidden-layer-100_epoch-50_batch_relu"
Path(modelname).mkdir(parents=True, exist_ok=True)
ann = Ann(dataset, 100, 50, False)
ann.AddLayer(12, ActivationFunctions.relu, dataset.GetNumberOfColumn() - 1)
ann.AddLayer(12, ActivationFunctions.relu, dataset.GetNumberOfColumn() - 1)
ann.AddLayer(12, ActivationFunctions.relu, dataset.GetNumberOfColumn() - 1)
ann.AddBinaryClassificationLayer(ActivationFunctions.sigmoid)
ann.BuildModel()

ann.PlotModel(
    f'./{modelname}/model.png')
#kfoldresult = ann.KFold(5)
# print(kfoldresult)

history, history_result = ann.QuickProcess(0.2)
ann.ExportModelAccuracyGraph(modelname, f'./{modelname}/')
print("MY HISTORY")
print(history_result)
field_names = ['precision', 'recall', 'f1-score', 'accuracy']
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




# last #9 value 
# 100 epoch, 50 batch size
# ann = Ann(dataset, 100, 50, False)
# # 2 hidden layer with 12 nodes, relu activation function
# ann.AddLayer(12, ActivationFunctions.relu, dataset.GetNumberOfColumn() - 1)
# ann.AddLayer(12, ActivationFunctions.relu, dataset.GetNumberOfColumn() - 1)
# ann.AddBinaryClassificationLayer(ActivationFunctions.sigmoid)
# ann.BuildModel()
# # 20% for test 80% training
# history, history_result = ann.QuickProcess(0.2)
# # select all data from dataset 
# x_test = dataset.GetXData()
# y_test = dataset.GetYData()
# # split for last 9 data
# last_9_x_test = x_test.tail(9)
# last_9_y_test = y_test.tail(9)
# scores = ann.EvaluateModel(last_9_x_test, last_9_y_test)
# print(f'Accuracy = {scores[1]}')
# print(ann.BinaryPredict(last_9_x_test))
