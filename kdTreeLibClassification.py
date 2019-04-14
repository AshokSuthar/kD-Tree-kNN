#make sure appropriate class column in getting compared with the end results. as datasets have different columns as class. some has last column as class, some have 3rd columns etc.
import numpy as np
import pandas as pd
import math
import time
import sys
from scipy import spatial
from collections import Counter

#function to generate 'num_points' random points of 'dim' dimensions.
def generate_data(filename):
	filename = sys.argv[1] #dataset to calculate coreset of
	#output = sys.argv[2] #output file to print probability distribution values

	if filename == "DataSets/bio_train.csv":
		dataset_df = pd.read_csv(filename,sep="\s+",header = None)
	elif filename == "DataSets/data_kddcup04/phy_train.dat":
		dataset_df = pd.read_csv(filename,sep="\s+",header = None)
	elif filename == "DataSets/MiniBoone.csv":
		dataset_df = pd.read_csv(filename,sep=",")
	elif filename == "DataSets/HTRU2/HTRU_2.xls":
		dataset_df = pd.read_excel(filename,sep=",",header = None)
	elif filename == "DataSets/shuttle/shuttle.xls":
		dataset_df = pd.read_excel(filename,sep="\s+",header = None)
	elif filename == "DataSets/default of credit card clients.xls":
		dataset_df = pd.read_excel(filename,sep="\s+",header = 0)
	elif filename == "DataSets/spambase/spambaseTrainTest.data":
		dataset_df = pd.read_csv(filename,sep=",",header = None)
	dim = dataset_df.shape[1]
	rows = dataset_df.shape[0]
	if filename == "DataSets/shuttle/shuttle.xls" or filename == "DataSets/MiniBoone.csv":
		data_df = dataset_df.iloc[:rows-10000, :dim] #full data with class values, removed more rows here to avoid maximum recursion limit.
	else:
		data_df = dataset_df.iloc[:rows-1000, :dim] #full data with class values
	return data_df


if __name__ == "__main__":
	print("Building the kdTree based on Meenawongtvana Paper (1999):")
	if len(sys.argv) != 2:
		print("use python3 programname.py <dataset_name> to run.")
		exit()
	filename = sys.argv[1] #dataset to calculate coreset of
	start_time = time.time()
	data_with_class = generate_data(filename) #dataset with class variables
	dim = data_with_class.shape[1]
	if filename == "DataSets/bio_train.csv":
		data = data_with_class.iloc[:,3:dim] #data without class variable
		df = pd.read_csv(filename,sep="\s+")
	elif filename == "DataSets/MiniBoone.csv":
		data = data_with_class.iloc[:,:dim-1] #data without class variable
		df = pd.read_csv(filename,sep=",")
	elif filename == "DataSets/HTRU2/HTRU_2.xls":
		data = data_with_class.iloc[:,:dim-1] #data without class variable
		df = pd.read_excel(filename,sep=",")
	elif filename == "DataSets/shuttle/shuttle.xls":
		data = data_with_class.iloc[:,:dim-1] #data without class variable
		df = pd.read_excel(filename,sep="\s+")
	elif filename == "DataSets/data_kddcup04/phy_train.dat":
		data = data_with_class.iloc[:,2:dim] #data without class variable
		df = pd.read_csv(filename,sep="\s+")
	elif filename == "DataSets/default of credit card clients.xls":
		data = data_with_class.iloc[:,1:dim-1] #data without class variable
		df = pd.read_excel(filename,sep="\s+")
	elif filename == "DataSets/spambase/spambaseTrainTest.data":
		data = data_with_class.iloc[:,:dim-1] #data without class variable
		df = pd.read_csv(filename,sep=",")
	dim = df.shape[1]
	rows = df.shape[0]
	leafsize= 50
	tree = spatial.KDTree(data, leafsize)
	#time in building index(offlinePhase)
	print("---time in building index(offlinePhase) %s seconds ---" % (time.time() - start_time))
	rightGuessCount = 0
	maxTime = -1000;
	minTime = 1000;
	totalTime = 0;
	for i in range(1,1000):
		query_point_with_class = df.iloc[rows-i:rows-(i-1), :dim] #query_point dataframe with class
		#building tree based on given points_list and leaf_size
		if filename == "DataSets/bio_train.csv":
			query_point = np.array(query_point_with_class.iloc[:,3:dim]) # using query_point without class variable
		elif filename == "DataSets/data_kddcup04/phy_train.dat":
			query_point = np.array(query_point_with_class.iloc[:,2:dim]) # using query_point without class variable
		elif filename == "DataSets/MiniBoone.csv":
			query_point = np.array(query_point_with_class.iloc[:,:dim-1]) # using query_point without class variable
		elif filename == "DataSets/HTRU2/HTRU_2.xls":
			query_point = np.array(query_point_with_class.iloc[:,:dim-1]) # using query_point without class variable
		elif filename == "DataSets/shuttle/shuttle.xls":
			query_point = np.array(query_point_with_class.iloc[:,:dim-1]) # using query_point without class variable
		elif filename == "DataSets/default of credit card clients.xls":
			query_point = np.array(query_point_with_class.iloc[:,1:dim-1]) # using query_point without class variable
		elif filename == "DataSets/spambase/spambaseTrainTest.data":
			query_point = np.array(query_point_with_class.iloc[:,:dim-1]) # using query_point without class variable
		#print("Data dimensions: "+str(data.shape))
		#starting time count
		start_time = time.time()
		k = 50
		dist,indices = (tree.query(query_point, k))
		#printing nearest neighbors
		#list of indices is indices[0]
		nnClassList = []
		#print("Nearest Points to the query are: ")
		for index in indices[0]:
			#change to appropriate class column based on the dataset
			if filename == "DataSets/bio_train.csv":
				nnClassList = np.hstack([nnClassList, np.array(data_with_class.iloc[index][2])])
			elif filename == "DataSets/data_kddcup04/phy_train.dat":
				nnClassList = np.hstack([nnClassList, np.array(data_with_class.iloc[index][1])]) #col 1 represents class here.
			else:
				nnClassList = np.hstack([nnClassList, np.array(data_with_class.iloc[index][dim-1])])
		#print(nnClassList)
		uniqw, inverse = np.unique(nnClassList, return_inverse=True)
		#print("unique inverse ",uniqw, inverse)
		arr = np.bincount(inverse)
		indexOfMaxOccur = np.where(arr == max(np.bincount(inverse)))
		newClass = uniqw[indexOfMaxOccur[0][0]] #indexOfMaxOccur is a list of one numpyArray with newClass as its first and only element. [0] accesses, numpy array and another [0] access actual index.
		#change to appropriate class column based on the dataset
		if filename == "DataSets/bio_train.csv":
			aClass = np.array(query_point_with_class)[0][2] 
		elif filename == "DataSets/data_kddcup04/phy_train.dat":
			aClass = np.array(query_point_with_class)[0][1] #col 1 represents class here.
		else:
			aClass = np.array(query_point_with_class)[0][dim-1]
		#print("Actual Class : ",aClass, " new Class: ",newClass)
		if aClass == newClass:
			rightGuessCount += 1
			#print("right ", rightGuessCount, "Times")
		#else:
			#print("WRONG WRONG WRONG WRONG WRONG WRONG WRONG WRONG WRONG")
		totalTime += (time.time() - start_time)
		if maxTime < (time.time() - start_time):
			maxTime = (time.time() - start_time)
		if minTime > (time.time() - start_time):
			minTime = (time.time() - start_time)
		#print("--- %s seconds ---" % ((time.time() - start_time)))
	print("RightGuesses: ", rightGuessCount, " MaxTime: ",maxTime, " MinTime: ",minTime, " AvgTime: ",totalTime/1000)
