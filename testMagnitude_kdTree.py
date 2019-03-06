import numpy as np
import pandas as pd
import math
import time

#class for internal nodes of the kdTree
#holds, cut_dim value at each node, along with cut_point itself for reference.
#left_child points to the left node of the tree, similarly right_child points to the right node of the tree.
class Inode:
	#cut_dim = 0
	cut_point = 0
	left_child = None
	right_child = None

	#init method for Inode class
	#initializes each node with cut_point, cut_dim, left_child and right_child of the node.
	def __init__(self,cut_point, left_child, right_child):
		self.cut_point = cut_point
		#self.cut_dim = cut_dim
		self.left_child = left_child
		self.right_child = right_child
	
	#__repr__ dundder method for fallback of __str__ dundder method
	def __repr__(self):
		return "{},{},{}".format(self.cut_point,self.left_child,self.right_child)

	#__str__ dundder method for printing the class instance in string format
	def __str__(self):
		return "node:{}, <------{},------>{}".format(self.cut_point,self.left_child,self.right_child)

#class for nodes holding leaf points of the kdTree
#points_list, which is all the points in that region(node) of the tree. tree division stops here.
class Lnode:
	points_list = None

	#init method for Lnode class
	#initializes each leaf node with points in that node.
	def __init__(self,points_list):
		self.points_list = points_list

	#__repr__ dundder method for fallback of __str__ dundder method
	def __repr__(self):
		return "{}".format(self.points_list)

	#__str__ dundder method for printing the class instance in string format
	def __str__(self):
		return "leaf_Points:{}".format(self.points_list)

def mag(point):
	point = np.array(point)
	return math.sqrt(np.sum(np.square(point)))

#KDTree function to build a Kd tree of a given points_list, cut_dim for dimension to split on, and max number of points a leaf can hold is given by leaf_size.
#splitting stops if a node has points less than the leaf_size
def KDTree(points_list, leaf_size):
	root = None
	cut_val = 0
	left_child = None
	right_child = None
	stop = False
	
	#If not enough points in the points_list -
	if(len(points_list) <= leaf_size):
		stop = True

	#this tree_node is a leaf node
	if stop == True:
		#leaf nodes added
		root = Lnode(points_list)

	#otherwise build kdTree recursively 
	else:
		#checking number of dimensions in data 
		#num_dimensions = len(points_list[0])
		mid = 0
		#sorting method is used for splitting here. 
		points_list = sorted(points_list, key = mag)
		#points_list.view('f8,f8').sort(order=['f'+str(cut_dim)], axis=0)
		#middle element from sorted list is picked as root to split upn
		mid = int(len(points_list) / 2)
		cut_point = points_list[mid].copy()
		#next_dimension to split upon
		#next_dim = (cut_dim + 1) % num_dimensions
		#recursive calls for the nodes(building kdTree) if not a leaf node
		root = Inode(cut_point, KDTree(points_list[:mid], leaf_size), KDTree(points_list[mid:], leaf_size))
	return root

#function to search for k nearest neighbors of a given query_point.
#accepts, root node, quer_point, and number of neighbors to search 'k' as parameters
def KNN(root, query_point, k):
	#checking if the node is an internal node, we only compare values according to cut_dim
	#if root.cut_point[axis] > query_point[axis] we go to the left_child of the tree
	#otherwise to the right_child and resume search
	if isinstance(root, Inode):
		if mag(root.cut_point) > mag(query_point):
			KNN(root.left_child, query_point, k)
		else:
			KNN(root.right_child, query_point, k)

	#otherwise if node is a leaf node, we search nearest node among the points_list of that leaf node
	else:
		# Creating a distance list from the leaf nodes
		closest_points = root.points_list
		closest_points = np.array(closest_points)
		query_point = np.array(query_point)
		distance = []
		#calculating distance of all the points in leaf node from the query_point and adding into distance list
		for i in range(len(closest_points)):
			#Calculating distance (2-d) and adding to list distance, along with the index of the point, i.
			distance.append([math.sqrt(np.sum(np.square(closest_points[i]-query_point))),i])
		distance = np.array(distance)
		#sorting the distance array based on the distances in the ascending order, while preserving their index(position) in points_list
		distance.view('f8,i8').sort(order=['f0'], axis=0)
		#print(distance)
		#print(str(k)+" Nearest points to the query point according to the M-KD_Tree are:")
		nn = []
		#taking only closest k points and adding into nearest_neighbor list
		for x in distance[:k]:
			#use print(x) to print distances of the closest k points
			#print(x)
			nn.append(closest_points[int(x[1])])
		#printing the knn
		#print(nn,end="\n\n")
		print(distance[:k])

#function to generate 'num_points' random points of 'dim' dimensions.
#function to generate 'num_points' random points of 'dim' dimensions.
def generate_data(data_type):
	if data_type == 1:
		df = pd.read_csv('DataSets/Lung.txt',sep="\s+",header=None)
		print("Taking Lung(181x12533) as input data")
		print(df.shape[1])
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
		df = pd.read_csv('DataSets/Random.txt',sep="\s+",header=None)
		data = df.iloc[100:500, :10000]
		print("Hey Taking Random("+str(data.shape)+") as input data")
		#converting to numpy array
		data = np.array(data)
	else:
		print("Generating normalized random data(1000x10000) as input data")
		data = np.random.rand(10000,1000)
	return data


#where execution starts
if __name__ == "__main__":
	print("Building the kdTree using Magnitude of pointVectors with given set of Points:")
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
	elif data_type == 5: 
		file_name = "query_random.txt"
		dim = 10000
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
	leaf_size = 20
	#checking runtime
	start_time = time.time()
	#starting Index Build (offline Phase)
	kd_tree = KDTree(data,leaf_size)
	print("---time in Index Building (Offline Phase) %s seconds ---" % (time.time() - start_time))
	#printing kdTree
	#print(kd_tree,end="\n\n")
	query_point = np.array(query_point)
	#number of neighbors to search for, value less then number of points in leaf
	#k = int(input("Enter the value of 'k'(less than "+str(leaf_size)+"):"))
	k=5
	#checking runtime
	start_time = time.time()
	#Querying for query_point
	KNN(kd_tree, query_point, k)
	print("--- %s seconds ---" % ((time.time() - start_time)))

	#making 1000 queries to compare time
	#start_time = time.time()
	#for _ in range(1000):
	#	dist = KNN(kd_tree, query_point, k)
	#print("--- 1000 Queries time = %s seconds ---" % ((time.time() - start_time)))











