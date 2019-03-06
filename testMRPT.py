import mrpt
import numpy as np
import pandas as pd
import math
import time
from scipy import spatial


start_time = time.time()
n, d, k = 200, 14096, 10
target_recall = 0.99

data_64 = np.random.rand(n, d)
df = pd.DataFrame(data_64)
print(df.describe())
#mrpt python binding requires float32
data_32 = data_64.astype(np.float32)

q_64 = np.random.rand(d)
#mrpt python binding requires float32
q_32 = q_64.astype(np.float32)

index = mrpt.MRPTIndex(data_32)
start_time = time.time()
print(index.exact_search(q_32, k, return_distances=True))

#index.build_autotune_sample(target_recall, k)
#start_time = time.time()
#print(index.ann(q, return_distances=True))
print("--- %s seconds ---" % (400*(time.time() - start_time)))







print("KdTreeLib MeenaWongtwana Paper 1999:")
#data_type = int(input("Choose appropriate input\n 1. Lung data set \n 2. Leukimia\n 3. GCM\n 4. Prostate \n 0. randomly generated data:\n"))
	#calling generate_data() for data to be generated/read.
#points_list = np.array(data)
#giving leaf size for the tree, to split further the tree should have more points than the leaf_size given here.
#leaf_size = int(input("Enter the value of leaf_size for the kD_Tree: "))
#building tree based on given points_list and leaf_size
tree = spatial.KDTree(data_64, leafsize=20)
#query_point = list(map(float,input("Enter the query point(same dimensions as datapoints): ").split()))
#number of neighbors to search for, value less then number of points in leaf
#k = int(input("Enter the value of 'k'(less than "+str(leaf_size)+"):"))
#query_point = 10*(np.random.rand(1,2))
#df = pd.read_csv('query_point.txt',sep="\s+",header=None)
#print(df)
#query_point = df.iloc[:, :7129]
#query_point = np.random.rand(10000).astype(np.float32)
#query_point = np.array(q)
start_time = time.time()
dist,indices = (tree.query(q_64, k))
#printing nearest neighbors
for index in indices:
	#print(tree.data[index])
	print(math.sqrt(np.sum(np.square(tree.data[index]-q_64))))
print("--- %s seconds ---" % (400*(time.time() - start_time)))





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

def dist(a,b):
	return math.sqrt(np.sum(np.square(a-b)))

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
	if isinstance(root.left_child, Inode) and isinstance(root.right_child,Inode):
		#if mag(root.cut_point) > mag(query_point):
		if dist(root.left_child.cut_point, query_point) < dist(root.right_child.cut_point,query_point):
			KNN(root.left_child, query_point, k)
		else:
			KNN(root.right_child, query_point, k)

	#otherwise if node is a leaf node, we search nearest node among the points_list of that leaf node
	else:
		if isinstance(root.left_child, Lnode):
			root = root.left_child
		else:
			root = root.right_child
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
		#print(str(k)+" Nearest points to the query point"+str(query_point)+"according to the KD_Tree built are:")
		nn = []
		#taking only closest k points and adding into nearest_neighbor list
		for x in distance[:k]:
			#use print(x) to print distances of the closest k points
			#print(x)
			nn.append(closest_points[int(x[1])])
		#printing the knn
		#print(nn,end="\n\n")
		print(distance[:k])


#checking runtime
start_time = time.time()

print("kdTree using Magnitude of pointVectors with given set of Points:")
#data_type = int(input("Choose appropriate input\n 1. Lung data set \n 2. Leukimia\n 3. GCM\n 4. Prostate \n 0. randomly generated data:\n"))
#calling generate_data() for data to be generated/read.
#data_type = 2
#data = generate_data(data_type)
#points_list = np.array(data)
#giving leaf size for the tree, to split further the tree should have more points than the leaf_size given here.
#leaf_size = int(input("Enter the value of leaf_size for the kD_Tree: "))
leaf_size = 20
kd_tree = KDTree(data_64,leaf_size)
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
#printing kdTree
#print(kd_tree,end="\n\n")
#query_point = list(map(float,input("Enter the query point(same dimensions as datapoints): ").split()))
#df = pd.read_csv('query_point.txt',sep="\s+",header=None)
#print(df)
#query_point = df.iloc[:, :7129]
#query_point = np.array(q)
#number of neighbors to search for, value less then number of points in leaf
#k = int(input("Enter the value of 'k'(less than "+str(leaf_size)+"):"))
KNN(kd_tree, q_64, k)
print("--- %s seconds ---" % (400*(time.time() - start_time)))







#checking runtime
start_time = time.time()

print("Brute Force")
distance = []
count = 0
for i in range(len(data_32)):
	count += 1
	#Calculating distance (2-d) and adding to list distance, along with the index of the point, i.
	distance.append([math.sqrt(np.sum(np.square(data_32[i]-q_32))),i])
distance = np.array(distance)
distance.view('f8,i8').sort(order=['f0'], axis=0)
print(count)
print(distance[:k])
print("--- %s seconds ---" % (400*(time.time() - start_time)))
