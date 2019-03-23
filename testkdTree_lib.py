import numpy as np
import pandas as pd
import math
import time
import sys
from scipy import spatial

#function to generate 'num_points' random points of 'dim' dimensions.
def generate_data(filename):
	filename = sys.argv[1] #dataset to calculate coreset of
	#output = sys.argv[2] #output file to print probability distribution values

	if filename == "DataSets/pd_speech_features.csv":
		dataset_df = pd.read_csv(filename,sep=",",header=[0,1])
		dim = dataset_df.shape[1]
		rows = dataset_df.shape[0]
		data_df = dataset_df.iloc[2:750, 2:dim-1]
	else:
		dataset_df = pd.read_csv(filename,sep="\s+",header=None)
		dim = dataset_df.shape[1]
		rows = dataset_df.shape[0]
		data_df = dataset_df.iloc[:rows-1, 2:dim]
	return np.array(data_df)



if __name__ == "__main__":
	print("Building the kdTree based on Meenawongtvana Paper (1999):")
	if len(sys.argv) != 2:
		print("use python3 programname.py <dataset_name> to run.")
		exit()
	filename = sys.argv[1] #dataset to calculate coreset of
	start_time = time.time()
	data = generate_data(filename)
	n,d,k = 10000,1000,5
	if filename == "DataSets/pd_speech_features.csv":
		dataset_df = pd.read_csv(filename,sep=",",header=[0,1])
		dim = dataset_df.shape[1]
		rows = dataset_df.shape[0]
		query_point = np.array(dataset_df.iloc[rows-1:rows, 2:dim-1])
	else:
		dataset_df = pd.read_csv(filename,sep="\s+",header=None)
		dim = dataset_df.shape[1]
		rows = dataset_df.shape[0]
		query_point = np.array(dataset_df.iloc[rows-1:rows, 2:dim])
	#building tree based on given points_list and leaf_size
	print("Data dimensions: "+str(data.shape))
	tree = spatial.KDTree(data, leafsize=20)
	#time in building index(offlinePhase)
	print("---time in building index(offlinePhase) %s seconds ---" % (time.time() - start_time))
	#starting time count
	start_time = time.time()
	dist,indices = (tree.query(query_point, k = 5))
	#printing nearest neighbors
	#list of indices is indices[0]
	for index in indices[0]:
		#print(tree.data[index])
		print(math.sqrt(np.sum(np.square(tree.data[index]-query_point))))
	print("--- %s seconds ---" % ((time.time() - start_time)))
	#start_time = time.time()
	#making 1000 queries
	#for _ in range(1000):
	#	dist,indices = (tree.query(query_point, k = 5))
		#list of indices is indices[0]
	#	for index in indices[0]:
			#print(tree.data[index])
	#		temp = math.sqrt(np.sum(np.square(tree.data[index]-query_point)))
	#print("---1000 Queries time =  %s seconds ---" % ((time.time() - start_time)))


