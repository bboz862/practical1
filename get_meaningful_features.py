import numpy as np
import csv
import math
from sklearn import feature_selection

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

meaningful_features_indice_filename = 'meaningful_features_indice.csv'
new_meaningful_features_indice_filename = 'new_meaningful_features_indice.csv'

read_molecule_data = read_compact_data('meaningful_features.csv')
molecular_features = read_molecule_data.get_data()
gaps = read_molecule_data.get_gap()

print molecular_features.shape

F_array, pval_array = feature_selection.f_regression(molecular_features, gaps)

with open(meaningful_features_indice_filename,'r') as indice_fh:
    indice_csv = csv.reader(indice_fh, delimiter=',', quotechar='"')
    for row in indice_csv:
    	meaningful_features_indice = row

new_meaningful_features_indice = []

for count in [0, 31]:

	if pval_array[count] < 0.05:
		new_meaningful_features_indice.append(count)

print new_meaningful_features_indice

with open(new_meaningful_features_indice_filename,'w') as indice_fh:
    indice_csv = csv.writer(indice_fh, delimiter=',', quotechar='"')
    indice_csv.writerow(meaningful_features_indice[new_meaningful_features_indice])


with open(new_meaningful_filename, 'w') as new_meaningful_fh:
    # Produce a CSV file.
    compact_csv = csv.writer(new_meaningful_fh, delimiter=',', quotechar='"')
    for datum in meaningful_data:
        compact_row = np.append(datum['features'][meaningful_features_indice[new_meaningful_features_indice]], datum['gap'])
        compact_csv.writerow(compact_row)

