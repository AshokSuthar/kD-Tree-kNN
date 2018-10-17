import numpy as np
import math

#class for internal nodes of the kdTree
#holds, cut_dim value at each node, along with cut_point itself for reference.
#left_child points to the left node of the tree, similarly right_child points to the right node of the tree.
class Inode:
	cut_dim = 0
	cut_point = 0
	left_child = None
	right_child = None

	#init method for Inode class
	#initializes each node with cut_point, cut_dim, left_child and right_child of the node.
	def __init__(self,cut_point, cut_dim, left_child, right_child):
		self.cut_point = cut_point
		self.cut_dim = cut_dim
		self.left_child = left_child
		self.right_child = right_child
	
	#__repr__ dundder method for fallback of __str__ dundder method
	def __repr__(self):
		return "{},{},{},{}".format(self.cut_point,self.cut_dim,self.left_child,self.right_child)

	#__str__ dundder method for printing the class instance in string format
	def __str__(self):
		return "node:{}, dim:{},<------{},------>{}".format(self.cut_point,self.cut_dim,self.left_child,self.right_child)

#class for nodes holding leaf points of the kdTree
#point_list, which is all the points in that region(node) of the tree. tree division stops here.
class Lnode:
	point_list = None

	#init method for Lnode class
	#initializes each leaf node with points in that node.
	def __init__(self,point_list):
		self.point_list = point_list

	#__repr__ dundder method for fallback of __str__ dundder method
	def __repr__(self):
		return "{}".format(self.point_list)

	#__str__ dundder method for printing the class instance in string format
	def __str__(self):
		return "leaf_Points:{}".format(self.point_list)

#KDTree function to build a Kd tree of a given point_list, cut_dim for dimension to split on, and max number of points a leaf can hold is given by leaf_size.
#splitting stops if a node has points less than the leaf_size
def KDTree(point_list, leaf_size, cut_dim=0):
	root = None
	cut_val = 0
	left_child = None
	right_child = None
	stop = False
	
	#If not enough points in the point_list -
	if(len(point_list) <= leaf_size):
		stop = True

	#this tree_node is a leaf node
	if stop == True:
		#leaf nodes added
		root = Lnode(point_list)

	#otherwise build kdTree recursively 
	else:
		#checking number of dimensions in data 
		num_dimensions = len(point_list[0])
		mid = 0
		#sorting method is used for splitting here. 
		point_list.view('f8,f8').sort(order=['f'+str(cut_dim)], axis=0)
		#middle element from sorted list is picked as root to split upn
		mid = int(len(point_list) / 2)
		cut_point = point_list[mid]
		#next_dimension to split upon
		next_dim = (cut_dim + 1) % num_dimensions
		#recursive calls for the nodes(building kdTree) if not a leaf node
		root = Inode(cut_point, cut_dim, KDTree(point_list[:mid], leaf_size, next_dim), KDTree(point_list[mid:], leaf_size, next_dim))
	return root

#function to search for k nearest neighbors of a given query_point.
#accepts, root node, quer_point, and number of neighbors to search 'k' as parameters
def KNN(root, query_point, k):
	#checking if the node is an internal node, we only compare values according to cut_dim
	#if root.cut_point[axis] > query_point[axis] we go to the left_child of the tree
	#otherwise to the right_child and resume search
	if isinstance(root, Inode):
		axis = root.cut_dim
		if root.cut_point[axis] > query_point[axis]:
			KNN(root.left_child, query_point, k)
		else:
			KNN(root.right_child, query_point, k)

	#otherwise if node is a leaf node, we search nearest node among the point_list of that leaf node
	else:
		# Creating a distance list from the leaf nodes
		closest_points = root.point_list
		distance = []
		#calculating distance of all the points in leaf node from the query_point and adding into distance list
		for i in range(len(closest_points)):
			#Calculating distance (2-d) and adding to list distance, along with the index of the point, i.
			distance.append([math.sqrt(pow(closest_points[i][0]-query_point[0],2)+pow(closest_points[i][1]-query_point[1],2)),i])
		distance = np.array(distance)
		#sorting the distance array based on the distances in the ascending order, while preserving their index(position) in point_list
		distance.view('f8,i8').sort(order=['f0'], axis=0)
		#print(distance)
		print(str(k)+" Nearest points to the query point"+str(query_point)+"according to the KD_Tree built are:")
		nn = []
		#taking only closest k points and adding into nearest_neighbor list
		for x in distance[:k]:
			#use print(x) to print distances of the closest k points
			#print(x)
			nn.append(closest_points[int(x[1])])
		#printing the knn
		print(nn,end="\n\n")

#function to generate random points in 2 dimensions. change to generate points in different dimensions or numbers.
def generate_random_points(num_points):
	points = []
	for _ in range(num_points):
		points.append((100*np.random.rand(), 100*np.random.rand()))
	return points

#where execution starts
if __name__ == "__main__":
	print("Building the kdTree with given set of Points:")
	#calling generate_random_points(num) for num of points to be generated.
	points_list = generate_random_points(10)
	points_list = np.array(points_list)
	#giving leaf size for the tree, to split further the tree should have more points than the leaf_size given here.
	leaf_size = 2
	kd_tree = KDTree(points_list,leaf_size)
	#printing kdTree
	print(kd_tree,end="\n\n")
	query_point = list(map(float,input("Enter the query point: ").split()))
	#number of neighbors to search for, value less then number of points in leaf
	k = int(input("Enter the value of 'k'(less than "+str(leaf_size)+"):"))
	KNN(kd_tree, query_point, k)
	













