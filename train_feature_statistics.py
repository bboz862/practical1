import csv
import numpy as np
import collections
import matplotlib.pyplot as plt

train_filename = 'train.csv'
#test_filename  = 'test.csv.gz'
compact_filename  = 'meaningful_features.csv'
meaningful_features_indice_filename = 'meaningful_features_indice.csv'

Load the training file.
train_data = []
with open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)

    # Load the data. For speed only load 100K samples.
    count = 1
    for row in train_csv:
        # if count >= 100000:
        #     break
        smiles   = row[0]
        features = np.array([float(x) for x in row[1:257]])
        gap      = float(row[257])
        
        train_data.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })
        # count += 1

# Compute the mean of the gaps in the training data.
gaps = np.array([datum['gap'] for datum in train_data])
feature_max_value = []
feature_min_value = []
for feature_number in range(0, 256):
    feature_max_value.append(max([dataitem['features'][feature_number] for dataitem in train_data]))
    feature_min_value.append(min([dataitem['features'][feature_number] for dataitem in train_data]))
# plt.hist(gaps,500)
# plt.show()
# print feature_max_value
# print feature_min_value
feature_value_difference = np.array(feature_max_value) - np.array(feature_min_value)
non_zero_features_number = np.count_nonzero(feature_value_difference)
non_zero_features_indice = np.nonzero(feature_value_difference)
# print non_zero_features_number
# print non_zero_features_indice
plt.plot(range(0,256),feature_max_value,'ro', range(0,256),feature_min_value,'g*')
plt.axis([0,256,-1,2])
plt.show()

#Load the test file.

# test_data = []
# with gzip.open(test_filename, 'r') as test_fh:
# 
#     # Parse it as a CSV file.
#     test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
#     
#     # Skip the header row.
#     next(test_csv, None)
# 
#     # Load the data.
#     for row in test_csv:
#         id       = row[0]
#         smiles   = row[1]
#         features = np.array([float(x) for x in row[2:258]])
#         
#         test_data.append({ 'id':       id,
#                            'smiles':   smiles,
#                            'features': features })
# 
# 
# # Write a prediction file.
#non_zero_features_indice = [0,4,5,6,24,36,43,67,68,71,86,89,101,118,122,125,131,172,175,186,195,198,199,207,217,224,225,242,247,250,251]
print non_zero_features_indice
with open(meaningful_features_indice_filename,'w') as indice_fh:
    indice_csv = csv.writer(indice_fh, delimiter=',', quotechar='"')
    indice_csv.writerow(non_zero_features_indice)

with open(compact_filename, 'w') as compact_fh:
    # Produce a CSV file.
    compact_csv = csv.writer(compact_fh, delimiter=',', quotechar='"')
    count = 1
    for datum in train_data:
        # if count >= 100000:
        #     break
        compact_row = np.append(datum['features'][non_zero_features_indice], datum['gap'])
        compact_csv.writerow(compact_row)
        # count += 1
