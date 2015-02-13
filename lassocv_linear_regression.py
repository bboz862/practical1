import numpy as np
import csv
import math
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import cluster
from multiprocessing import Process

class read_compact_data:

	def __init__(self, filename):
		
		with open(filename, 'r') as fileheader:
		    csv_file = csv.reader(fileheader, delimiter=',', quotechar='"')

		    csv_data = []
		    
		    for row in csv_file:
		    	features = np.array([float(x) for x in row[0:275]])
		        energy_gap      = float(row[275])		        
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
		self._data_cluster = cluster.MiniBatchKMeans(n_clusters = self._basis_number, batch_size = 1000)
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

def main(basis_number = 100, CV_candidates = None, CV_folds = 10):
	
	#read in training data
	read_molecule_data = read_compact_data('meaningful_features_2_512_10000.csv')
	molecular_features = read_molecule_data.get_data()

	#get shape of training data
	feature_number = molecular_features.shape[1]
	sample_number = molecular_features.shape[0]
	print 'There are {} features and {} samples in training data'.format(feature_number, sample_number)

	#get training target value
	molecule_energygap = read_molecule_data.get_gap()

	#generate parameters for training basis functions
	basis_parameter = find_radial_basis_function(molecular_features, basis_number)
	center_vector = basis_parameter.get_center()
	width_square_vector = basis_parameter.get_width_square()

	#transform input data based on training basis functions
	transformed_data_matrix = radial_basis_function_transformation(molecular_features, center_vector, width_square_vector)
	print 'After transformation by radial basis functions the training data now has dimensions of {}'.format(transformed_data_matrix.shape)

	#generate stacked output of basis functions (including identity basis)
	transformed_molecular_features = np.hstack([molecular_features, transformed_data_matrix])
	print 'Plus the identity basis the training data has dimensions of {}'.format(transformed_molecular_features.shape)

	#do Lasso regression
	Lasso_regression = linear_model.LassoCV(alphas =  CV_candidates, max_iter = 10000, n_jobs = len(CV_candidates), cv = cross_validation.KFold(sample_number, n_folds = CV_folds))
	Lasso_regression.fit(transformed_molecular_features, molecule_energygap)
	
	# print Lasso_regression.coef_
	print 'The penalty given by the regression algorithm is {}'.format(Lasso_regression.alpha_)

	#calculate prediction accuracy on training data
	predicted_train_gap_energy = Lasso_regression.predict(transformed_molecular_features)
	average_train_error = math.sqrt( np.dot( (predicted_train_gap_energy - molecule_energygap), (predicted_train_gap_energy - molecule_energygap).T ) / float(sample_number) )
	print 'On the training data set the average error of prediction is {}'.format(average_train_error)

	#get data and target value test set
	read_test_data = read_compact_data('meaningful_features_2_512_10000-15000.csv')
	test_molecular_features = read_test_data.get_data()
	test_molecular_energygap = read_test_data.get_gap()
	test_sample_number = test_molecular_features.shape[0]

	#transform test data based on learned basis function
	transformed_test_matrix = radial_basis_function_transformation(test_molecular_features, center_vector, width_square_vector)
	print 'After transformation by radial basis functions the testing data now has dimensions of {}'.format(transformed_test_matrix.shape)

	#generate stacked output of basis functions
	transformed_test_features = np.hstack([test_molecular_features, transformed_test_matrix])
	print 'Plus the identity basis the testing data has dimensions of {}'.format(transformed_test_features.shape)

	#make prediction on test set
	predicted_test_gapenergy = Lasso_regression.predict(transformed_test_features)
	average_test_error = math.sqrt( np.dot( (predicted_test_gapenergy - test_molecular_energygap), (predicted_test_gapenergy - test_molecular_energygap).T ) / float(test_sample_number) )
	print 'On the testing data set the average error of prediction is {}'.format(average_test_error)

if __name__ == '__main__': 
	
	p = Process(target = main, args = ( 500, [ 1e-6, 1e-5, 1e-4 ]))
	p.start()
	p.join()