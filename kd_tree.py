import numpy as np
import math

class Inode:
	cut_dim = 0
	cut_point = 0
	left_child = None
	right_child = None

	def __init__(self,cut_point, cut_dim, left_child, right_child):
		self.cut_point = cut_point
		self.cut_dim = cut_dim
		self.left_child = left_child
		self.right_child = right_child
	
	def __repr__(self):
		return "{},{},{},{}".format(self.cut_point,self.cut_dim,self.left_child,self.right_child)

	def __str__(self):
		return "node:{}, dim:{},<------{},------>{}".format(self.cut_point,self.cut_dim,self.left_child,self.right_child)

class Lnode:
	point_list = None

	def __init__(self,point_list):
		self.point_list = point_list

	def __repr__(self):
		return "{}".format(self.point_list)

	def __str__(self):
		return "leaf_Points:{}".format(self.point_list)

def KDTree(point_list, cut_dim=0, leaf_size=5):
	root = None
	cut_val = 0
	left_child = None
	right_child = None
	stop = False
	
	if(len(point_list) <= leaf_size):
		# Not enough points in the list - this tree is a leaf node
		stop = True
	if stop == True:
		#leaf nodes added
		root = Lnode(point_list)
		#print(root)
	else:
		num_dimensions = len(point_list[0])
		mid = 0
		#print("Before")
		#print(point_list)
		if cut_dim == 0:
			point_list.view('f8,f8').sort(order=['f0'], axis=0)
		else:
			point_list.view('f8,f8').sort(order=['f1'], axis=0)
		#point_list.sort(axis = cut_dim)
		#print("After")
		#print(point_list)
		mid = int(len(point_list) / 2)
		cut_point = point_list[mid]
		#print(root)
		#cut_val = point_list[mid][cut_dim]
		#print(cut_val)
		next_dim = (cut_dim + 1) % num_dimensions
		#recursive calls
		#print("<----")
		#left_child = KDTree(point_list[:mid], next_dim, leaf_size)
		#print("---->")
		#right_child = KDTree(point_list[mid:], next_dim, leaf_size)
		#no leaf nodes
		root = Inode(cut_point, cut_dim, KDTree(point_list[:mid], next_dim, leaf_size), KDTree(point_list[mid:], next_dim, leaf_size))
	return root

def KNN(root, query_point, k):
	if isinstance(root, Inode):
		axis = root.cut_dim
		if root.cut_point[axis] > query_point[axis]:
			KNN(root.left_child, query_point, k)
		else:
			KNN(root.right_child, query_point, k)
	else:
		# Creating a distance list from the leaf nodes
		closest_points = root.point_list
		distance = []
		for i in range(len(closest_points)):
			#Calculating distance (2-d) and adding to list distance, along with the index of the point, i.
			distance.append([math.sqrt(pow(closest_points[i][0]-query_point[0],2)+pow(closest_points[i][1]-query_point[1],2)),i])
		distance = np.array(distance)
		#print(distance)
		#sorting the distance array based on the distances
		distance.view('f8,i8').sort(order=['f0'], axis=0)
		#print(distance)
		print(str(k)+" Nearest points to the query point"+str(query_point)+"according to the KD_Tree built are:")
		#taking only closest k points.
		nn = []
		for x in distance[:k]:
			print(x)
			nn.append(closest_points[int(x[1])])
		print(nn)
	




def generate_random_points(num_points):
	points = []
	for _ in range(num_points):
		points.append((100*np.random.rand(), 100*np.random.rand()))
	return points

if __name__ == "__main__":
	print("Trying to run final KdTree")
	points_list = generate_random_points(10)
	points_list = np.array(points_list)
	leaf_size = 5
	kd_tree = KDTree(points_list,leaf_size=2)
	print(kd_tree)
	print()
	query_point = list(map(float,input("Enter the query point: ").split()))
	#number of neighbors to search for, value less then number of points in leaf
	k = int(input("Enter the value of 'k'(less than "+str(leaf_size)+"):"))
	KNN(kd_tree, query_point, k)
	print()
	













