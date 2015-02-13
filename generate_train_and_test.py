import csv

long_filename = 'meaningful_features.csv'

train_filename = 'meaningful_features_100000.csv'
test_filename  = 'meaningful_features_100000-200000.csv'

input_file        = open(long_filename,'r')
output_train_file = open( train_filename,'w')
output_test_file  = open( test_filename,'w')


input_csv = csv.reader(input_file, delimiter=',', quotechar='"')
train_csv = csv.writer(output_train_file, delimiter=',', quotechar='"')
test_csv = csv.writer(output_test_file, delimiter=',', quotechar='"')

count = 1 
for row in input_csv:
	if count > 200000:
		break
	elif count <= 100000:
		train_csv.writerow(row)
	elif count <= 200000:
		test_csv.writerow(row)
	count += 1

input_file.close
output_train_file.close
output_test_file.close

