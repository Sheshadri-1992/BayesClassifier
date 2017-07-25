from matplotlib.pyplot import *
import csv
from math import *
import operator
import math
import random
import k_cross_validation


def loadSamples(fileName):
	# file_name = "census-income.data"
	data =[]
	train=[]

	with open(fileName) as f:
		reader = csv.reader(f)
		# data = list(list(record) for record in  csv.reader(f,delimiter=','))
		data = []
		for record in csv.reader(f,delimiter=','):
			data.append(record)
		f.close()

	
	continuous = [0,5,16,17,18,24,30,39]

	for item in data:
		
		datarow = []
		for i in range(len(item)):

			if i not in continuous:
				datarow.append(item[i])

		train.append(datarow)

	return train





dataset = "census-income.data"
samples = []
samples = loadSamples(dataset)

accuracyList=[]
num_epoch = 30 

print "Naive-Bayes' classifier"
K = 10
for epochs in range(num_epoch):

	random.shuffle(samples) # data is shuffled now, we can perform 10 cross validation now
	
	epochAccuracy = 0
	for i in range(K):		
		epochAccuracy = epochAccuracy + k_cross_validation.prepare_train_test_data(samples,i,K)

	epochAccuracy = epochAccuracy / float(K)
	print "epoch: ",epochs,", Accuracy: ",epochAccuracy
	accuracyList.append(epochAccuracy)



accuracyList=np.array(accuracyList)
print "Mean ",accuracyList.mean()
print "Standard Deviation ",np.std(accuracyList) 
