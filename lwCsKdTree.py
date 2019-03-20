import numpy as np
import pandas as pd
import math
import sys
import time
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn import datasets

clusters = {}
centroids = []
np_data = []

#function to generate 'num_points' random points of 'dim' dimensions.
def generate_data(filename, m):
	#if data_type == 1:
	#	pass
	#filename = sys.argv[1] #dataset to calculate coreset of
	#output = sys.argv[2] #output file to print probability distribution values

	dataset_df = pd.read_csv(filename,sep="\s+",header = None)

	dim = dataset_df.shape[1]
	rows = dataset_df.shape[0]
	data_df = dataset_df.iloc[:rows-1, 1:dim]
	class_df = dataset_df.iloc[:rows-1, 0:1]
	print(class_df)
	#print(data_df.head())
	rows = data_df.iloc[:] #all the rows in selected dataset
	data_size = len(rows)#calculating #no. of entries in data(no. of rows)
	data = np.array(data_df)
	data_mean = np.mean(data, axis = 0)
	print(data_mean)
	distance = 0
	for point in data:
		distance += np.sum(np.square(point-data_mean))
	print(distance)
	prob_dist = []
	#calculating proposal  distribution for each row
	for point in data:
		value=((0.5*(1/data_size))+0.5*((np.sum(np.square(point-data_mean)))/distance))
		prob_dist.append(value)
	df = pd.DataFrame(prob_dist)
	#print(prob_dist)
	data_df['Prob_dist'] = df
	#print(data_df)
	#writing ProbDist to file
	#dataset.to_csv("//home//oseen//Documents//Mtech_project//code2_KDDdata2004//light_bio_train1.csv",index=False)
	#adding weight value to dataset
	weight_value = []
	for i in range(data_size):
		#print("i is: ",i," m is ",m,"prob_dist is ",prob_dist[0]," type is ",prob_dist[0].dtype)
		weight = 1/(m*prob_dist[i])
		weight_value.append(weight)
	df = pd.DataFrame(weight_value)
	data_df['weight_value'] = df
	class_df['weight_value'] = df
	print(class_df)
	#dataset.to_csv("//home//oseen//Documents//Mtech_project//code2_KDDdata2004//light_bio_train1.csv",index=False)
	#sorting result
	sorted_data = data_df.sort_values('weight_value',ascending='False')
	sorted_class = class_df.sort_values('weight_value',ascending='False')
	print(sorted_data.iloc[:m,:dim-1])
	print(sorted_class.iloc[:m,:dim-1])
	print(sorted_class.iloc[13])
	return sorted_data.iloc[:m,:dim-1]



if __name__ == "__main__":
	#calling generate_data() for data to be generated/read.
	if len(sys.argv) != 3:
		print("use python3 programname.py <dataset_name> <size of coreset 'm'>to run.")
		exit()
	filename = sys.argv[1] #dataset to calculate coreset of
	m = int(sys.argv[2])
	start_time = time.time()
	data = generate_data(filename, m)
	n,d,k = 10000,1000,5
	df = pd.read_csv(filename,sep="\s+")
	dim = df.shape[1]
	rows = df.shape[0]
	query_point = np.array(df.iloc[rows-1:rows, 1:dim])
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
		print(tree.data[index])
	print(query_point)
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

