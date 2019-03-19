import numpy as np
import pandas as pd
import math
import time
import sys
from scipy import spatial

#function to generate 'num_points' random points of 'dim' dimensions.
def generate_data(filename):
	dataset_df = pd.read_csv(filename,sep="\s+",header=None)
	dim = dataset_df.shape[1]
	rows = dataset_df.shape[0]
	data_df = dataset_df.iloc[:rows-1, 2:dim]
	data = np.array(data_df)
	return data



if __name__ == "__main__":
	print("Building the kdTree based on Meenawongtvana Paper (1999):")
	#calling generate_data() for data to be generated/read.
	if len(sys.argv) != 2:
		print("use python3 programname.py <dataset_name> to run.")
		exit()
	filename = sys.argv[1]
	start_time = time.time()
	data = generate_data(filename)
	df = pd.read_csv(filename,sep="\s+",header=None)
	print("shape of the input data",df.shape)
	dim = df.shape[1]
	rows = df.shape[0]
	query_point = df.iloc[rows-1:rows, 2:dim]
	query_point = np.array(query_point)
	print("Query Point shape "+str(query_point.shape))
			
	#if file_name != None:
	#	df = pd.read_csv(file_name, sep="\s+", header=None)
	#	query_point = df.iloc[:, :dim]
	#	query_point = np.array(query_point)
	#else:
	#	data = np.random.rand(n,d)
	#	query_point = np.random.rand(d)
	#giving leaf size for the tree, to split further the tree should have more points than the leaf_size given here.
	#leaf_size = int(input("Enter the value of leaf_size for the kD_Tree: "))
	#starting time count
	#start_time = time.time()
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


