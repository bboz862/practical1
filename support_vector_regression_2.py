import numpy as np
import gzip
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

class read_test_compact_data:

	def __init__(self, filename):
		
		with open(filename, 'r') as fileheader:
		    csv_file = csv.reader(fileheader, delimiter=',', quotechar='"')

		    csv_data = []
		    
		    for row in csv_file:
		    	features = np.array([float(x) for x in row[0:275]])
		        iden      = float(row[275])		        
		        csv_data.append({'features':      features,
		                            'id':   iden })

		self._data_array = np.array([dict_item['features'] for dict_item in csv_data])
		self._id_array = np.array([dict_item['id'] for dict_item in csv_data])

		del csv_data

	def get_data(self):
		return self._data_array

	def get_id(self):
		return self._id_array

def main():
	#read in training data
	read_molecule_data = read_compact_data('meaningful_features_2_512_50000.csv')
	molecular_features = read_molecule_data.get_data()

	#get shape of training data
	feature_number = molecular_features.shape[1]
	sample_number = molecular_features.shape[0]
	print 'There are {} features and {} samples in the training data'.format(feature_number, sample_number)

	#get training target value
	molecule_energygap = read_molecule_data.get_gap()

	#do support_vector regression
	support_vector_regression = svm.SVR( C = 1, gamma = 0.05, epsilon = 0.1, cache_size = 5000)
	support_vector_regression.fit(molecular_features, molecule_energygap)
	
	# print support_vector_regression.coef_

	#calculate prediction accuracy on training data
	predicted_train_gap_energy = support_vector_regression.predict(molecular_features)
	average_train_error = math.sqrt( np.dot( (predicted_train_gap_energy - molecule_energygap), (predicted_train_gap_energy - molecule_energygap).T ) / float(sample_number) )
	print 'On the training data set the average error of prediction is {}'.format(average_train_error)

	#get data and target value test set
	read_test_data = read_test_compact_data('test_meaningful_features_2_512.csv')
	test_molecular_features = read_test_data.get_data()
	test_molecular_id = read_test_data.get_id()
	test_sample_number = test_molecular_id.shape[0]

	#make prediction on test set
	predicted_test_gapenergy = support_vector_regression.predict(test_molecular_features)

	pred_filename = 'prediction_svr_2.csv'

	# Write a prediction file.
	with open(pred_filename, 'w') as pred_fh:
		pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')
		pred_csv.writerow(['Id', 'Prediction'])
		for n in range(test_sample_number):
			pred_csv.writerow( [test_molecular_id[n], predicted_test_gapenergy[n]] )


if __name__ == '__main__': 
	
	p = Process( target = main )
	p.start()