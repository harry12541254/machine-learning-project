"""
File: interactive.py
Name: 
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
import graderUtil
import util
from util import *
from submission import *

def main():
	filename = './weights'
	weight = {}
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip().split()
			num = float(line[1])
			weight[line[0]] = num
	interactivePrompt(extractWordFeatures, weight)



if __name__ == '__main__':
	main()
