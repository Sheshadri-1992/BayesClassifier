#!/usr/bin/python
from sklearn import svm
import sys
import numpy as np
from sklearn.svm import SVC
import time
import matplotlib.pyplot as py

# for a particular value of a feature say "windy", get how many times it is present 
# in a given class
def getFeatureCount(mainDict,class_,feature,featureVal):

	if (featureVal in mainDict[class_][feature]) and (featureVal != ' ?'):
		return mainDict[class_][feature][featureVal]
	else:
		return 0

#create the dictionary structure to save frequencies of all components, per class
def storeDataInMemory(samples):

	myDict = {}
	for row in samples:
		class_ = row[-1]

		if class_ not in myDict:
			myDict[class_] = {}

		for i in range(len(row)-1):
			if i not in myDict[class_]:
				myDict[class_][i] = {}

			featureVal = row[i]
			if featureVal not in myDict[class_][i]:
				myDict[class_][i][featureVal] = 1
			else:
				myDict[class_][i][featureVal] = myDict[class_][i][featureVal] + 1

	return myDict		

#pass sample dataset , train the model with train data set, test on testDataset
def trainAndClassifyNaiveBayes(train_data,test_data,classCount,mainDict):

	#classCount: class1 -> 100, class2->170 
	classNames = classCount.keys()
	# print "classNames ",classNames
	class1 = classNames[0]
	class2 = classNames[1]

	totalTrain = classCount[class1] + classCount[class2]
	
	#apriori probabilities
	p_w1 = classCount[class1]/float(totalTrain)
	p_w2 = classCount[class2]/float(totalTrain)

	accuracy = 0

	#classify each test data row
	for row in test_data:

		post_w1_x = 1
		post_w2_x = 1
		
		#multiply probabilites for all individual feature components
		for i in range(len(row)-1): #number of feature components

			#for class1
			p_x_w1 = getFeatureCount(mainDict,class1,i,row[i])/float(classCount[class1])
			post_w1_x = post_w1_x * (p_x_w1)

			#for class2
			p_x_w2 = getFeatureCount(mainDict,class2,i,row[i])/float(classCount[class2])
			post_w2_x = post_w2_x * (p_x_w2)

		#final formula: posterior = (class conditional * apriori)
		post_w1_x = post_w1_x * p_w1
		post_w2_x = post_w2_x * p_w2

		actualLabel = row[-1] # last entry in the data set is the class label

		if post_w1_x > post_w2_x : #class 1 posterior greater

			predictedLabel = class1			
			if (predictedLabel == actualLabel):
				accuracy = accuracy + 1

		elif post_w2_x > post_w1_x : #class 2 posterior greater

			predictedLabel = class2		
			if (predictedLabel == actualLabel):
				accuracy = accuracy + 1

		else : # if both are equal then decide on apriori probability

			if (p_w1 > p_w2) : #class 1 apriori is greater than class2

				predictedLabel = class1			
				if (predictedLabel == actualLabel):
					accuracy = accuracy + 1

			else: #class 2 apriori is greater than class1

				predictedLabel = class2			
				if (predictedLabel == actualLabel):
					accuracy = accuracy + 1				


	accuracy = accuracy / float(len(test_data)) ;
	
	return accuracy

# return dictionary of class frequencies
def getClassFrequency(train_data):
	
	countClass={}
	for item in train_data:

		class_ = item[-1]
		if class_ not in countClass:
			countClass[class_]=1
		else:
			countClass[class_]=countClass[class_]+1

	return countClass	

def prepare_train_test_data(k_cross_samples,skip_number,k):
	
	train_data = []
	test_data = []
	
	
	for i in range(len(k_cross_samples)):
		if( i%k == skip_number):
			test_data.append(k_cross_samples[i])
		else:
			train_data.append(k_cross_samples[i])

	mainDict = {}
	mainDict = storeDataInMemory(train_data)

	classCount = getClassFrequency(train_data) #classCount has count of rows of class1 and class2

	accuracy = trainAndClassifyNaiveBayes(train_data,test_data,classCount,mainDict)

	del train_data[:]
	del test_data[:]

	return accuracy
