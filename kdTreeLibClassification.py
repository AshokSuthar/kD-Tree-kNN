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

	if filename == "DataSets/bio_train.csv":
		dataset_df = pd.read_csv(filename,sep="\s+",header = None)
	else:
		dataset_df = pd.read_excel(filename,sep="\s+",header = 0)
	dim = dataset_df.shape[1]
	rows = dataset_df.shape[0]
	data_df = dataset_df.iloc[:rows-10, :dim] #full data with class values
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
	else:
		data = data_with_class.iloc[:,1:dim-1] #data without class variable
		df = pd.read_excel(filename,sep="\s+")
	dim = df.shape[1]
	rows = df.shape[0]
	query_point_with_class = df.iloc[rows-2:rows-1, :dim] #query_point dataframe with class
	#building tree based on given points_list and leaf_size
	if filename == "DataSets/bio_train.csv":
		query_point = np.array(query_point_with_class.iloc[:,3:dim]) # using query_point without class variable
	else:
		query_point = np.array(query_point_with_class.iloc[:,1:dim-1]) # using query_point without class variable
	#print("Data dimensions: "+str(data.shape))
	tree = spatial.KDTree(data, leafsize=3)
	#time in building index(offlinePhase)
	print("---time in building index(offlinePhase) %s seconds ---" % (time.time() - start_time))
	#starting time count
	start_time = time.time()
	dist,indices = (tree.query(query_point, k = 3))
	#printing nearest neighbors
	#list of indices is indices[0]
	print("Nearest Points to the query are: ")
	for index in indices[0]:
		#print(tree.data[index])
		#print(tree.data[index])
		print(np.array(data_with_class.iloc[index]))
	print("Query_point is: ")
	print(np.array(query_point_with_class))
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

