import numpy as np
import csv
import math
from sklearn import feature_selection

train_filename = 'train.csv'
new_meaningful_features_indice_filename = 'new_meaningful_features_indice.csv'
new_meaningful_filename = 'new_meaningful_features.csv'

train_data = []
with open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)

    # Load the data. For speed only load 100K samples.
    for row in train_csv:
        smiles   = row[0]
        features = np.array([float(x) for x in row[1:257]])
        gap      = float(row[257])
        
        train_data.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })

# molecular_features = np.array([dataitem['features'] for dataitem in train_data])
# gaps = np.array([dataitem['gap'] for dataitem in train_data])

new_meaningful_features_indice = np.array([0,5,6,24,36,43,67,68,71,86,89,101,118,122,125,131,172,175,186,195,198,199,207,217,224,225,242,247,250,251])

print new_meaningful_features_indice.shape

with open(new_meaningful_features_indice_filename,'w') as indice_fh:
    indice_csv = csv.writer(indice_fh, delimiter=',', quotechar='"')
    indice_csv.writerow(new_meaningful_features_indice)


with open(new_meaningful_filename, 'w') as new_meaningful_fh:
    # Produce a CSV file.
    compact_csv = csv.writer(new_meaningful_fh, delimiter=',', quotechar='"')
    for datum in train_data:
        compact_row = np.append(datum['features'][new_meaningful_features_indice], datum['gap'])
        compact_csv.writerow(compact_row)

