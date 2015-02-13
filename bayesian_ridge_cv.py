import numpy as np
import csv
import math
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import cluster
from multiprocessing import Process

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
		self._data_cluster = cluster.MiniBatchKMeans(n_clusters = self._basis_number, batch_size = 50000)
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

def main( alpha_1 = 1e-6, alpha_2 = 1e-6, lambda_1 = 1e-6, lambda_2 = 1e-6, basis_number = 300):

	cv_train_error_vector = []
	cv_test_error_vector = []
	cv_error_difference = []

	#read in training data
	read_molecule_data = read_compact_data('meaningful_features_200000.csv')
	molecular_features = read_molecule_data.get_data()

	#get shape of training data
	feature_number = molecular_features.shape[1]
	sample_number = molecular_features.shape[0]
	print 'For alpha_1 = {}, alpha_2 = {}, lambda_1 = {}, lambda_2 = {} there are {} features and {} samples in the training data'.format(alpha_1, alpha_2, lambda_1, lambda_2, feature_number, sample_number)

	#get training target value
	molecule_energygap = read_molecule_data.get_gap()

	basis_parameter = find_radial_basis_function(molecular_features, basis_number)
	center_vector = basis_parameter.get_center()
	width_square_vector = basis_parameter.get_width_square()

	#transform input data based on training basis functions
	transformed_data_matrix = radial_basis_function_transformation(molecular_features, center_vector, width_square_vector)

	#generate stacked output of basis functions (including identity basis)
	transformed_molecular_features = np.hstack([molecular_features, transformed_data_matrix])

	#generate cross validataion iterator 
	kf_cv = cross_validation.KFold(n = 200000, n_folds = 5)

	#iterate using cross validation iterator
	for train_index_array, test_index_array in kf_cv:

		cv_train_data = transformed_molecular_features[train_index_array]
		cv_train_gapenergy = molecule_energygap[train_index_array]
		cv_test_data = transformed_molecular_features[test_index_array]
		cv_test_gapenergy = molecule_energygap[test_index_array]

		#train the support vector machine
		bayesian_ridge_regression = linear_model.BayesianRidge( n_iter = 5000, alpha_1 = alpha_1, alpha_2 = alpha_2, lambda_1 = lambda_1, lambda_2 = lambda_2)
		bayesian_ridge_regression.fit(cv_train_data, cv_train_gapenergy)

		#calculate mean error in training data
		predicted_cv_train_gapenergy = bayesian_ridge_regression.predict(cv_train_data)
		average_cv_train_error = math.sqrt( np.dot( (predicted_cv_train_gapenergy - cv_train_gapenergy), (predicted_cv_train_gapenergy - cv_train_gapenergy).T ) / float(160000) )
		cv_train_error_vector.append(average_cv_train_error)

		#calculate mean error in test data
		predicted_cv_test_gapenergy = bayesian_ridge_regression.predict(cv_test_data)
		average_cv_test_error = math.sqrt( np.dot( (predicted_cv_test_gapenergy - cv_test_gapenergy), (predicted_cv_test_gapenergy - cv_test_gapenergy).T ) /  float(40000))
		cv_test_error_vector.append(average_cv_test_error)

		#calculate the difference between training error and test error
		cv_error_difference.append(average_cv_test_error - average_cv_train_error)

	average_train_error =  np.average(np.array(cv_train_error_vector))
	average_test_error = np.average(np.array(cv_test_error_vector))
	average_error_difference = math.sqrt( np.dot( np.array(cv_error_difference), np.array(cv_error_difference).T) / 5.0 )

	print 'For alpha_1 = {}, alpha_2 = {}, lambda_1 = {}, lambda_2 = {} on the training data set the average error of prediction is {}'.format( alpha_1, alpha_2, lambda_1, lambda_2, average_train_error )

	print 'For alpha_1 = {}, alpha_2 = {}, lambda_1 = {}, lambda_2 = {} on the testing data set the average error of prediction is {}'.format(alpha_1, alpha_2, lambda_1, lambda_2, average_test_error)

	print 'For alpha_1 = {}, alpha_2 = {}, lambda_1 = {}, lambda_2 = {} the average error difference between training data and testing data is {}'.format(alpha_1, alpha_2, lambda_1, lambda_2, average_error_difference)


if __name__ == '__main__': 
	for lambda_1 in [1e-7, 1e-6, 1e-5]:
		for lambda_2 in [1e-7, 1e-6, 1e-5]:
			p = Process( target = main, args = ( 1e-6, 1e-6, lambda_1, lambda_2) )
			p.start()

	