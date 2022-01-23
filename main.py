#!/usr/bin/env python3
import argparse
import logging
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

dataset = DataSet(arguments.csv, arguments.index, False)
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

print(ActivationFunctions.IsExist(ActivationFunctions, ActivationFunctions.elu))
print(ActivationFunctions.GetName(ActivationFunctions, ActivationFunctions.elu))
ann = Ann(dataset, 5, 32, False)
ann.AddLayer(20, ActivationFunctions.relu, dataset.GetNumberOfColumn() - 1)
ann.BuildModel()
ann.PlotModel(
    "/media/yakup/Samsung980/freelance/ml-interface/ml-interface/firstmodel.png")
ann.BuildModel()
ann.AddLayer(15, ActivationFunctions.relu)
ann.AddLayer(18, ActivationFunctions.relu)
ann.AddLayer(18, ActivationFunctions.relu)

ann.PlotModel(
    "/media/yakup/Samsung980/freelance/ml-interface/ml-interface/secondmodel.png")
ann.AddBinaryClassificationLayer(ActivationFunctions.sigmoid)
ann.PlotModel(
    "/media/yakup/Samsung980/freelance/ml-interface/ml-interface/binaryclass.png")
kfoldresult = ann.KFold(5)
print(kfoldresult)
