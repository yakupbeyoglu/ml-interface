#!/usr/bin/env python3
import argparse
import logging


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
