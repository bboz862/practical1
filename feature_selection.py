import numpy as np
import csv
import math
from sklearn import svm
from sklearn import cross_validation
from sklearn import cluster
from sklearn import feature_selection
from multiprocessing import Process

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

def main():
	read_molecule_data = read_compact_data('meaningful_features.csv')
	
	molecular_features = read_molecule_data.get_data()

	molecular_energygap = read_molecule_data.get_gap()

	F_array, pval_array = feature_selection.f_regression(molecular_features, molecular_energygap)

	print F_array

	print pval_array

if __name__ == '__main__': main()