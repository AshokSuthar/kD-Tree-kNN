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
	elif data_type == 5:
		df = pd.read_csv('DataSets/bio_train.dat',sep="\s+",header=None)
		print("Taking bio_train data as input data")
		dim = df.shape[1]
		data = df.iloc[:, 3:dim]
		#converting to numpy array
		data = np.array(data)
	elif data_type == 6:
		df = pd.read_csv('DataSets/pd_speech_features.csv',sep = ",",header=[0,1])
		print("Taking pd_speech_features data as input data")
		dim = df.shape[1]
		data = df.iloc[2:750, 2:dim-1]
		print(data.describe())
		#converting to numpy array
		data = np.array(data)
		print(data.dtype)
		print(data.shape)
	else:
		print("Generating normalized random data(1000x10000) as input data")
		data = np.random.rand(10000,1000)
	return data



if __name__ == "__main__":
	print("Building the kdTree based on Meenawongtvana Paper (1999):")
	data_type = int(input("Choose appropriate input\n 1. Lung data set \n 2. Leukimia\n 3. GCM\n 4. Prostate \n 0. randomly generated data:\n"))
	#calling generate_data() for data to be generated/read.
	start_time = time.time()
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
	elif data_type == 5: 
		df = pd.read_csv('DataSets/bio_test.dat',sep="\s+",header=None)
		dim = df.shape[1]
		query_point = df.iloc[:1, 3:dim]
		query_point = np.array(query_point)
	elif data_type == 6: 
		df = pd.read_csv('DataSets/pd_speech_features.csv',sep=",",header=[0,1])
		dim = df.shape[1]
		query_point = df.iloc[752:753, 2:dim-1]
		query_point = np.array(query_point)
		print("Query Point shape "+str(query_point.shape))
	else:
		file_name = None
		
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
	#tree = spatial.KDTree(data, leafsize=20)
	#time in building index(offlinePhase)
	print("---time in building index(offlinePhase) %s seconds ---" % (time.time() - start_time))
	#starting time count
	start_time = time.time()
	print("Brute Force")
	distance = []
	count = 0
	for i in range(len(data)):
		count += 1
		#Calculating distance (2-d) and adding to list distance, along with the index of the point, i.
		distance.append([math.sqrt(np.sum(np.square(data[i]-query_point))),i])
	distance = np.array(distance)
	distance.view('f8,i8').sort(order=['f0'], axis=0)
	print(count)
	print(distance[:k])
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



start_time = time.time()


print("--- %s seconds ---" % (400*(time.time() - start_time)))