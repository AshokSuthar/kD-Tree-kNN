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
def generate_data(filename):
	#if data_type == 1:
	#	pass
	#filename = sys.argv[1] #dataset to calculate coreset of
	#output = sys.argv[2] #output file to print probability distribution values

	dataset_df = pd.read_csv(filename,sep="\s+",header=None)
	dim = dataset_df.shape[1]
	rows = dataset_df.shape[0]
	data_df = dataset_df.iloc[:rows-1, 2:dim]
	#print(data_df.head())
	#print(x)
	rows = data_df.iloc[:] #all the rows in selected dataset
	data_size = len(rows)#calculating #no. of entries in data(no. of rows)
	no_clusters = 20# change this
	print(no_clusters)
	global np_data #using global np_data variable
	np_data = np.array(data_df) #converting data to numpy array
	estimator = KMeans(n_clusters=no_clusters)
	estimator.fit(np_data)
	global clusters 
	clusters = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}
	points = []
	global centroids
	for key in clusters.keys():
		for index in clusters[key]:
			points.append(np_data[index])
		#print(np.array(points))
		centroids.append(np.array(points).mean(axis=0))
	centroids = np.array(centroids)
	data = centroids
	return data



if __name__ == "__main__":
	#calling generate_data() for data to be generated/read.
	filename = sys.argv[1] #dataset to calculate coreset of
	start_time = time.time()
	data = generate_data(filename)
	n,d,k = 10000,1000,5
	df = pd.read_csv(filename,sep="\s+",header=None)
	dim = df.shape[1]
	rows = df.shape[0]
	query_point = np.array(df.iloc[rows-1:rows, 2:dim])
		
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
	print("data dimensions: "+str(data.shape))
	tree = spatial.KDTree(data, leafsize=2)
	#time in building index(offlinePhase)
	print("---time in building index(offlinePhase) %s seconds ---" % (time.time() - start_time))
	#starting time count
	start_time = time.time()
	dist,indices = (tree.query(query_point, k = 2))
	#finding which cluster this nearest_point belongs to.
	values = []
	#print(centroids)
	#for index in indices[0]:
	#	print(tree.data[index])
	#finding which centroid points came as nearest neighbors for the query points and adding them to values list.
	#list of indices of nearest points is indices[0]
	print(indices[0])
	for index in indices[0]:
		values.append(tree.data[index]) #tree.data is the array of all nearest neighbors to query point.
	cluster_indices = (np.where(np.isin(centroids[:,1], values))) #using the value list in finding which cluster these centroid points represent or belong to. So that we can do a in depth search in that cluster
	points = []
	for index in cluster_indices[0]: #finding index of all points belonging to this nearest cluster.
		point_indices = (clusters[index]) #indices of all points belonging to this cluster
		for point_index in point_indices: 
			points.append(np_data[point_index]) #adding all points in the said cluster to the list "points".
	
	distance = []
	for i in range(len(points)):
	#Calculating distance (2-d) and adding to list "distanc", along with the index(in the cluster) of the point, i.
		distance.append([math.sqrt(np.sum(np.square(points[i]-query_point))),i])
	distance = np.array(distance)
	distance.view('f8,i8').sort(order=['f0'], axis=0)#sorting all the distances of points in ascending order.
	print(distance[:k]) #printing 'k' nearest neighbors

	print("--- %s seconds ---" % ((time.time() - start_time)))
	#print(dist)
	#start_time = time.time()
	#making 1000 queries
	#for _ in range(1000):
	#	dist,indices = (tree.query(query_point, k = 5))
		#list of indices is indices[0]
	#	for index in indices[0]:
			#print(tree.data[index])
	#		temp = math.sqrt(np.sum(np.square(tree.data[index]-query_point)))
	#print("---1000 Queries time =  %s seconds ---" % ((time.time() - start_time)))





