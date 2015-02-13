
# coding: utf-8

# In[1]:

import csv
import gzip
import numpy as np
from sklearn import linear_model as lm
from sklearn import svm

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'example_mean.csv'


# In[ ]:

# Load the training file.
train_data = []
with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)

    # Load the data.
    for row in train_csv:
        smiles   = row[0]
        features = np.array([float(x) for x in row[1:257]])
        gap      = float(row[257])
        
        train_data.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })


# In[45]:

clf = lm.Lasso()
clf.fit([[1,1,1],[1,0,1],[0,0,1]],[0,0,1])
clf.coef_


# In[26]:

# Compute the mean of the gaps in the training data.
gaps = np.array([datum['gap'] for datum in train_data])
features = [datum['features'] for datum in train_data]
clf = lm.Ridge(alpha= 1.0)
clf.fit(features,gaps)


# In[27]:

features[:5]


# In[ ]:




# In[4]:

# Load the test file.
test_data = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    for row in test_csv:
        id       = row[0]
        smiles   = row[1]
        features = np.array([float(x) for x in row[2:258]])
        
        test_data.append({ 'id':       id,
                           'smiles':   smiles,
                           'features': features })


# In[31]:

# Compute the mean of the gaps in the training data.
features = [datum['features'] for datum in test_data]
pred = clf.predict(features)
pred[:100]


# In[33]:

# Write a prediction file.
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    for x in xrange(0,len(test_data)):
        pred_csv.writerow([test_data[x]['id'], pred[x]])


# In[28]:

gaps[1]


# max(gaps)

# In[44]:

min(gaps)


# In[21]:

gaps[:10]


# In[30]:

[datum['features'] for features in test_data{:10}]


# In[ ]:



