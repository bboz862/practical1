import numpy as np
import csv
import math
from sklearn import svm
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

def main( C = 1, gamma = 0.03, epsilon = 0.1):

	cv_train_error_vector = []
	cv_test_error_vector = []
	cv_error_difference = []

	#read in training data
	read_molecule_data = read_compact_data('meaningful_features_2_512_50000.csv')
	molecular_features = read_molecule_data.get_data()

	#get shape of training data
	feature_number = molecular_features.shape[1]
	sample_number = molecular_features.shape[0]
	print 'For C = {}, gamma = {}, epsilon = {}, there are {} features and {} samples in the training data'.format(C, gamma, epsilon,feature_number, sample_number)

	#get training target value
	molecule_energygap = read_molecule_data.get_gap()

	#generate cross validataion iterator 
	kf_cv = cross_validation.KFold(n = 50000, n_folds = 5)

	#iterate using cross validation iterator
	for train_index_array, test_index_array in kf_cv:

		cv_train_data = molecular_features[train_index_array]
		cv_train_gapenergy = molecule_energygap[train_index_array]
		cv_test_data = molecular_features[test_index_array]
		cv_test_gapenergy = molecule_energygap[test_index_array]

		#train the support vector machine
		support_vector_regression = svm.SVR( C = C, gamma = gamma, epsilon = epsilon, cache_size = 1000)
		support_vector_regression.fit(cv_train_data, cv_train_gapenergy)

		#calculate mean error in training data
		predicted_cv_train_gapenergy = support_vector_regression.predict(cv_train_data)
		average_cv_train_error = math.sqrt( np.dot( (predicted_cv_train_gapenergy - cv_train_gapenergy), (predicted_cv_train_gapenergy - cv_train_gapenergy).T ) / float(40000) )
		cv_train_error_vector.append(average_cv_train_error)

		#calculate mean error in test data
		predicted_cv_test_gapenergy = support_vector_regression.predict(cv_test_data)
		average_cv_test_error = math.sqrt( np.dot( (predicted_cv_test_gapenergy - cv_test_gapenergy), (predicted_cv_test_gapenergy - cv_test_gapenergy).T ) /  float(10000))
		cv_test_error_vector.append(average_cv_test_error)

		#calculate the difference between training error and test error
		cv_error_difference.append(average_cv_test_error - average_cv_train_error)

	average_train_error =  np.average(np.array(cv_train_error_vector))
	average_test_error = np.average(np.array(cv_test_error_vector))
	average_error_difference = math.sqrt( np.dot( np.array(cv_error_difference), np.array(cv_error_difference).T) / 5.0 )

	print 'For C = {}, gamma = {}, epsilon = {}, on the training data set the average error of prediction is {}'.format( C, gamma, epsilon, average_train_error )

	print 'For C = {}, gamma = {}, epsilon = {}, On the testing data set the average error of prediction is {}'.format(C, gamma, epsilon, average_test_error)

	print 'For C = {}, gamma = {}, epsilon = {}, the average error difference between training data and testing data is {}'.format(C, gamma, epsilon, average_error_difference)


if __name__ == '__main__': 
	for gamma in [0.01, 0.05, 0.1, 0.5, 1]:
		p = Process( target = main, args = ( 1, gamma, 0.1) )
		p.start()

	