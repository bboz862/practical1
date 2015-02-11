import numpy as np
import csv
import math
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import cluster

# define a class to extract molecular feature and energy gap between SUMO and HOMO
class read_compact_data:

	def __init__(self, filename):
		
		with open(filename, 'r') as fileheader:
		    csv_file = csv.reader(fileheader, delimiter=',', quotechar='"')

		    csv_data = []
		    
		    for row in csv_file:
		    	features = np.array([float(x) for x in row[0:31]])
		        energy_gap      = float(row[31])		        
		        csv_data.append({'features':      features,
		                            'energy_gap':   energy_gap })

		self._data_array = np.array([dict_item['features'] for dict_item in csv_data])
		self._energy_gap_array = np.array([dict_item['energy_gap'] for dict_item in csv_data])

		del csv_data

	def get_data(self):
		return self._data_array

	def get_gap(self):
		return self._energy_gap_array

#define a class to find center and radius of radial basis functions using K-Means clustering
class find_radial_basis_function:

	#trian K-Means with data
	def __init__(self, data, basis_number, neighbour_num = 2):
		self._basis_number = basis_number
		self._neighbour_num = neighbour_num
		self._data_cluster = cluster.MiniBatchKMeans(n_clusters = self._basis_number, batch_size = 10000)
		self._data_cluster.fit(data)

	#return centers of basis functions
	def get_center(self):
		self._center_array = self._data_cluster.cluster_centers_
		return self._center_array

	#calculate and return square of radius based on nearest neighbour method
	def get_width_square(self):
		distance_square_matrix = np.array([[ np.dot((x-y), (x-y).T) for x in self._center_array] for y in self._center_array])
		#print distance_square_matrix
		sorted_distance_square_matrix = np.sort(distance_square_matrix)
		#print sorted_distance_square_matrix
		self._width_square = np.array([np.sum( x[1 : (self._neighbour_num + 1)])/float(self._neighbour_num) \
			for x in sorted_distance_square_matrix])
		return self._width_square

def radial_basis_function_transformation(data, centers, width_square):
	distance_square_matrix = np.array([[np.dot((x - mu), (x - mu).T) for mu in centers] for x in data])
	scaled_distance_square_matrix = np.divide(distance_square_matrix, width_square)
	transformed_data_matrix = np.exp(-scaled_distance_square_matrix)
	return transformed_data_matrix

def main(basis_number = 100, CV_candidates = [0.01, 0.1, 1],CV_folds = 10):
	
	read_molecule_data = read_compact_data('meaningful_features_100000.csv')
	molecular_features = read_molecule_data.get_data()

	feature_number = molecular_features.shape[1]
	sample_number = molecular_features.shape[0]
	print feature_number, sample_number

	molecule_energygap = read_molecule_data.get_gap()

	basis_parameter = find_radial_basis_function(molecular_features, basis_number)
	center_vector = basis_parameter.get_center()
	width_square_vector = basis_parameter.get_width_square()

	transformed_data_matrix = radial_basis_function_transformation(molecular_features, center_vector, width_square_vector)
	print transformed_data_matrix.shape

	transformed_molecular_features = np.hstack([molecular_features, transformed_data_matrix])
	print transformed_molecular_features.shape

	ridge_regression = linear_model.RidgeCV(alphas =  CV_candidates, cv = cross_validation.KFold(sample_number, n_folds = CV_folds))
	ridge_regression.fit(transformed_molecular_features, molecule_energygap)
	
	# print ridge_regression.coef_
	print ridge_regression.alpha_

	pridected_gap_energy = ridge_regression.predict(transformed_molecular_features)
	average_error = math.sqrt( np.dot( (pridected_gap_energy - molecule_energygap), (pridected_gap_energy - molecule_energygap).T ) / float(sample_number) )
	print average_error

if __name__ == '__main__': main(basis_number = 200, CV_candidates = [ 1.125, 1.15, 1.175 ])


