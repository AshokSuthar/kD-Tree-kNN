import numpy as np
import pandas as pd
import math
import time
from scipy import spatial

#function to generate 'num_points' random points of 'dim' dimensions.
def generate_data(data_type):
	if data_type == 1:
		df = pd.read_csv('DataSets/Lung.txt',sep="\s+",header=None)
		print("Taking Lung(181x12533) as input data")
		data = df.iloc[:, :12533]
		#converting to numpy array
		data = np.array(data)
	elif data_type == 2:
		df = pd.read_csv('DataSets/Leukimia.txt',sep="\s+",header=None)
		print("Taking Leukimia(72x7129) as input data")
		data = df.iloc[:, :7129]
		#converting to numpy array
		data = np.array(data)
	elif data_type == 3:
		df = pd.read_csv('DataSets/GCM.txt',sep="\s+",header=None)
		print("Taking GCM(280x16064) as input data")
		data = df.iloc[:, :16064]
		#converting to numpy array
		data = np.array(data)
	elif data_type == 4:
		df = pd.read_csv('DataSets/Prostate.txt',sep="\s+",header=None)
		print("Taking Prostate(181x12600) as input data")
		data = df.iloc[:, :12600]
		#converting to numpy array
		data = np.array(data)
	else:
		print("Generating normalized random data(1000x10000) as input data")
		data = np.random.rand(10000,1000)
	return data



if __name__ == "__main__":
	print("Building the kdTree based on Meenawongtvana Paper (1999):")
	data_type = int(input("Choose appropriate input\n 1. Lung data set \n 2. Leukimia\n 3. GCM\n 4. Prostate \n 0. randomly generated data:\n"))
	#calling generate_data() for data to be generated/read.
	data = generate_data(data_type)
	n,d,k = 10000,1000,5
	if data_type == 1: 
		file_name = "query_point_Lung.txt"
		dim = 12533
	elif data_type == 2: 
		file_name = "query_point_Leukimia.txt"
		dim = 7129
	elif data_type == 3: 
		file_name = "query_point_GCM.txt"
		dim = 280
	elif data_type == 4: 
		file_name = "query_point_Prostate.txt"
		dim = 12600
	else:
		file_name = None
		
	if file_name != None:
		df = pd.read_csv(file_name, sep="\s+", header=None)
		query_point = df.iloc[:, :dim]
		query_point = np.array(query_point)
	else:
		data = np.random.rand(n,d)
		query_point = np.random.rand(d)
	#giving leaf size for the tree, to split further the tree should have more points than the leaf_size given here.
	#leaf_size = int(input("Enter the value of leaf_size for the kD_Tree: "))
	#starting time count
	start_time = time.time()
	#building tree based on given points_list and leaf_size
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


