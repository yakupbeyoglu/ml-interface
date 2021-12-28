#!/usr/bin/env python3
import argparse
import logging
from ML.DataSet import DataSet
from ML.GausianNaiveBayes import GausianNaiveBayes
from ML.LogisticRegression import LogisticRegressionModel
argparser = argparse.ArgumentParser(
    description='Run this software with csv path and class index in the csv file'
)

argparser.add_argument('--csv', type=str, help='CSV file path')
argparser.add_argument('--index', type=int, help='Index of the class in the csv file')

arguments = argparser.parse_args()
if not arguments.csv :
    raise argparse.ArgumentTypeError('csv is not given')

if not arguments.index : 
    raise argparse.ArgumentTypeError('--index is not given')

print(f'csv path : {arguments.csv} \nindex of the class in csv : {arguments.index}')

dataset = DataSet(arguments.csv, arguments.index, False);
dataset.GetBarChartOfClasses()
x,y = dataset.GetXYData()
print(x)
print("Hey you can find values below\n", dataset.GetAmountOfClasses())

gausian_naive_bayes = GausianNaiveBayes(dataset)

print("below : \n\n\n")
values = gausian_naive_bayes.KFold(5)
print("Naive bayes KFOLD 5 :\n" , values)
regression = LogisticRegressionModel(dataset)
values = regression.KFold(5)
print("Logistic Regression KFOLD 5:\n", values)

values = regression.KBest(5, 5)
print("Best Feature = \n", values)
print(gausian_naive_bayes.DataSetSummarize())
zeros, ones = gausian_naive_bayes.separate_by_class()
zeros, ones = gausian_naive_bayes.DataSetSummarize(zeros,ones)
print("zeros", zeros)
gausian_naive_bayes.GetProbabilityOfColumn("age", 50, zeros,ones)
